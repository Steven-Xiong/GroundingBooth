import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data import ConcatDataset as data_concat
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from convert_ckpt import add_additional_channels
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
try:
    from apex import amp
except:
    pass  

from dataset.mvimgnet import MVImageNet_Grounding, draw_box
from dataset.lvis import LvisDataset
from dataset.coco import COCODataset
from dataset.openimages import OpenImagesDatasetWithMask,OpenImagesDataset, OpenImagesDatasetWithMask_train
# from dataset.YoutubeVOS import YouTubeVOSDataset
from dataset.ref_ytvos import YTVOSDataset,make_coco_transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torchvision import transforms
from dataset.data_transfer import prepare_batch_hetero, get_clip_feature,project
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #

def my_collate_fn(batch):
    # batch 是一个列表，里面每个元素是 __getitem__ 的返回值
    # 这里过滤掉为 None 的数据
    batch = [sample for sample in batch if sample is not None]

    # 如果 batch 全部是 None，会变成空列表，需要做一下特殊处理
    if len(batch) == 0:
        # 这里可以视情况返回空张量，或者干脆 return None
        return None

    # 如果剩下的 batch 不为空，可以直接用 PyTorch 默认的拼接方式
    return torch.utils.data.dataloader.default_collate(batch)

class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, masked_real, ref,  captions, seen):
        # import pdb; pdb.set_trace()  # images.shape/real.shape: [B,3,512,512]
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_ref.png')
        torchvision.utils.save_image( ref, save_path, nrow=self.nrow)
        
        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 



class Trainer:
    def __init__(self, config):
        # import pdb; pdb.set_trace()
        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)
        # add image encoder:
        # import pdb; pdb.set_trace()
        self.image_encoder = instantiate_from_config(config.image_encoder).to(self.device)
        self.image_encoder_global = instantiate_from_config(config.image_encoder_global).to(self.device)

        # import pdb; pdb.set_trace()
        state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        
        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels>0 else False

        # load original SD ckpt (with inuput conv may be modified) 
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
        # add image encoder
        # state_dict_image = 
        # self.image_encoder.load_state_dict()
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)
        # import pdb; pdb.set_trace()
        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")  # '/project/osprey/scratch/x.zhexiao/GLIGEN/gligen_checkpoints/checkpoint_generation_text_image.bin'
            self.model.load_state_dict(first_stage_ckpt["model"], strict = False)


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        # import pdb; pdb.set_trace()
        for name, p in self.model.named_parameters():
            if ("transformer_blocks" in name) and ("fuser" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            # elif ("transformer_blocks" in name) and ("attn1" in name):  # add self attention training
            #     # New added Attention layers 
            #     params.append(p) 
            #     trainable_names.append(name)
            
            # elif ("transformer_blocks" in name) and ("attn2" in name):  # add cross attention training
            #     # New added Attention layers 
            #     params.append(p) 
            #     trainable_names.append(name)
                
            elif ("transformer_blocks" in name) and ("attn3" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            
            elif ("transformer_blocks" in name) and ("ff" in name):
                # feed forward layers 
                params.append(p) 
                trainable_names.append(name)
                
            elif  "position_net" in name:
                # Grounding token processing network 
                # import pdb; pdb.set_trace()
                params.append(p) 
                trainable_names.append(name)
            # elif  "image_encoder_global" in name:
            #     # Grounding token processing network 
            #     import pdb; pdb.set_trace()
            #     params.append(p) 
            #     trainable_names.append(name)

            elif  "downsample_net" in name:
                # Grounding downsample network (used in input) 
                params.append(p) 
                trainable_names.append(name)
            elif (self.input_conv_train) and ("input_blocks.0.0.weight" in name):
                # First conv layer was modified, thus need to train 
                params.append(p) 
                trainable_names.append(name)
            else:
                # Following make sure we do not miss any new params
                # all new added trainable params have to be haddled above
                # otherwise it will trigger the following error  
                assert name in original_params_names, name 
            all_params_name.append(name) 
        
        for name, p in self.image_encoder.named_parameters():
            if ("projector" in name):
                # import pdb; pdb.set_trace()
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
                print('image encoder training name:', name)
            all_params_name.append(name)

        for name, p in self.image_encoder_global.named_parameters():
            if ("projector" in name):
                # import pdb; pdb.set_trace()
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
                print('image encoder global training name:', name)
            all_params_name.append(name)

        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
        # import pdb; pdb.set_trace()
        
        version = "openai/clip-vit-large-patch14"
        self.clip_model = CLIPModel.from_pretrained(version).cuda()
        self.processor = CLIPProcessor.from_pretrained(version)
        self.transform_to_pil = transforms.ToPILImage()
        # import pdb; pdb.set_trace()
        self.get_clip_feature = get_clip_feature(model=self.clip_model, processor=self.processor,input=None, is_image=True)
        self.projection_matrix = torch.load('projection_matrix').cuda()
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 
        
        # mvimgnet单独训练,加lvis
        if config.use_mvimgnet == True:
            # dataset_train =  MVImageNet_Grounding( txt='/project/osprey/scratch/x.zhexiao/GLIGEN/data/MVImgNet_full/list.txt', image_dir='/project/osprey/scratch/x.zhexiao/GLIGEN/data/MVImgNet_full/') 
            # dataset_train = LvisDataset(image_dir= 'data/lvis_v1/train2017', json_path='data/lvis_v1/lvis_v1_train.json')
            
            # dataset_train_mvimgnet =  MVImageNet_Grounding( txt='data/MVImgNet_full/list.txt', image_dir='data/MVImgNet_full/') 
            # dataset_train_lvis = LvisDataset(image_dir= 'data/lvis_v1/train2017', json_path='data/lvis_v1/lvis_v1_train.json')
            # dataset_train_lvis1 = LvisDataset(image_dir= 'data/lvis_v1/train2017', json_path='data/lvis_v1/lvis_v1_train.json')
            # dataset_train = data_concat([dataset_train_mvimgnet, dataset_train_lvis,dataset_train_lvis1])
            
            # finetune on coco?
            # import pdb; pdb.set_trace()
            # dataset_train = COCODataset(image_dir= 'data/coco/images/train2017', json_path='data/coco/annotations/instances_train2017.json')
            # 2.10 openimages dataset
            # dataset_train_youtubevos = YouTubeVOSDataset(image_dir='data/YouTubeVOS/train', anno='data/YouTubeVOS/annotations', meta = 'data/YoutubeVOS/valid/meta.json')
            # dataset_train_YTVOS = YTVOSDataset(img_folder = 'data/ref_youtubevos/train', ann_file = 'data/ref_youtubevos/meta_expressions/train/meta_expressions.json', \
            #                                     transforms=make_coco_transforms('train', max_size=640), \
            #                                     num_frames=5, max_skip=3) 
            # # dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks, 
            # #                num_frames=args.num_frames, max_skip=args.max_skip)
            
            # dataset_train = dataset_train_YTVOS
            # dataset_train_YTVOS = YTVOSDataset(img_folder = 'data/ref_youtubevos/train', ann_file = 'data/ref_youtubevos/meta_expressions/train/meta_expressions.json', \
            #                                     transforms=make_coco_transforms('train', max_size=640), \
            #                                     num_frames=10, max_skip=3) 
            # dataset_train = dataset_train_YTVOS
            # dataset_train.getitem(1)
            
            ###### train on openimages_train   #####
            # dataset_train_mvimgnet =  MVImageNet_Grounding( txt='data/MVImgNet_full/list.txt', image_dir='data/MVImgNet_full/') 
            # # dataset_train_mvimgnet1 =  MVImageNet_Grounding( txt='data/MVImgNet_full/list.txt', image_dir='data/MVImgNet_full/') 
            # dataset_train_lvis = LvisDataset(image_dir= 'data/lvis_v1/train2017', json_path='data/lvis_v1/lvis_v1_train.json')
            # dataset_train_openimages = OpenImagesDatasetWithMask(data_root='data/Open_Imagesv7',tokenizer = self.tokenizer, set="train")
            # # dataset_train_openimages = OpenImagesDatasetWithMask_train(data_root='data/Open_Imagesv7',tokenizer = self.tokenizer, set="train")
            
            # dataset_train_YTVOS = YTVOSDataset(img_folder = 'data/ref_youtubevos/train', ann_file = 'data/ref_youtubevos/meta_expressions/train/meta_expressions.json', \
            #                                     transforms=make_coco_transforms('train', max_size=640), \
            #                                     num_frames=10, max_skip=3) 
            # dataset_train_YTVOS1 = YTVOSDataset(img_folder = 'data/ref_youtubevos/train', ann_file = 'data/ref_youtubevos/meta_expressions/train/meta_expressions.json', \
            #                                     transforms=make_coco_transforms('train', max_size=640), \
            #                                     num_frames=10, max_skip=3) 
            # dataset_train = data_concat([dataset_train_mvimgnet, dataset_train_lvis,dataset_train_openimages,dataset_train_YTVOS, dataset_train_YTVOS1])

            
            
            dataset_train_mvimgnet =  MVImageNet_Grounding( txt='data/MVImgNet_full/list.txt', image_dir='data/MVImgNet_full/') 
            dataset_train_lvis = LvisDataset(image_dir= 'data/lvis_v1/train2017', json_path='data/lvis_v1/lvis_v1_train.json')
            dataset_train_openimages = OpenImagesDatasetWithMask(data_root='data/Open_Imagesv7',tokenizer = self.tokenizer, set="test")
            # dataset_train_openimages = OpenImagesDatasetWithMask_train(data_root='data/Open_Imagesv7',tokenizer = self.tokenizer, set="train")
            
            dataset_train_YTVOS = YTVOSDataset(img_folder = 'data/ref_youtubevos/train', ann_file = 'data/ref_youtubevos/meta_expressions/train/meta_expressions.json', \
                                                transforms=make_coco_transforms('train', max_size=640), \
                                                num_frames=10, max_skip=3) 
            dataset_train = data_concat([dataset_train_mvimgnet,dataset_train_lvis,dataset_train_openimages,dataset_train_YTVOS])
            
            
        # 要改dataloader在这里改
        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        else:
            train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
            dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        # dataset_train.getitem(1)
        # import pdb; pdb.set_trace()
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle= (sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler,
                                                   collate_fn=my_collate_fn)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            # total_image = dataset_train.total_images()
            # print("Total training images: ", total_image)     
            print("Total training images: ", len(dataset_train))  
        



        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        # import pdb; pdb.set_trace()
        if checkpoint is not None:
            print('loading checkpoint from:', checkpoint)
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            # add all params
            self.image_encoder.load_state_dict(checkpoint["image_encoder"])
            self.image_encoder_global.load_state_dict(checkpoint["image_encoder_global"])
            self.text_encoder.load_state_dict(checkpoint["text_encoder"])  #这两个是不训的
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])
            self.diffusion.load_state_dict(checkpoint["diffusion"])

            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters : #+100000
                synchronize()
                print("Training finished. Start exiting")
                exit()


        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        # import pdb; pdb.set_trace()
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False, find_unused_parameters=True ) # True会报错
            self.model._set_static_graph()
        # import pdb; pdb.set_trace()
        


    @torch.no_grad()
    def get_input(self, batch):     #这里比较关键
        # import pdb; pdb.set_trace()
        # print(batch.keys())

        if self.config.use_mvimgnet:
            # image_embeddings = []
            # for i in range(batch['jpg'].shape[0]):
            #     ref = batch['ref_processed'][i]
            #     # ref_embedding = self.get_clip_feature(model=self.clip_model, processor=self.processor, input=ref.permute(2,0,1), is_image=True)
            #     ref_embedding = get_clip_feature(model=self.clip_model, processor=self.processor, input=ref.permute(2,0,1), is_image=True)
            #     image_embeddings.append(torch.cat((ref_embedding, torch.zeros((9, 768)).cuda()), dim=0)) 
            # batch['image_embeddings'] = torch.stack(image_embeddings, dim=0)   
            # TODO: 将每一个text prompt提取对应的text embedding
            # import pdb; pdb.set_trace()
            text_embeddings = []
            image_embeddings = []
            # list的维度其实是[]
            for tags in batch['positive_all']:
                text_inbatch_embeddings = []
                image_inbatch_embeddings = []
                tag_list = tags.split(',')
                for tag in tag_list:
                    # import pdb;pdb.set_trace()
                    inputs = self.processor(text=tag,return_tensors="pt", padding=True)  # batch['positive']是对应的object caption
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['pixel_values'] = batch['ref_processed'].cuda() # we use our own preprocessing without center_crop 
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    outputs = self.clip_model(**inputs)     
                    text_before_features = outputs.text_model_output.pooler_output # before projection feature
                    image_after_features = outputs.image_embeds

                    text_inbatch_embeddings.append(text_before_features)
                    image_inbatch_embeddings.append(image_after_features)
                # import pdb; pdb.set_trace()
                text_embeddings.append(torch.cat(text_inbatch_embeddings,dim=0).unsqueeze(0))
                image_embeddings.append(torch.cat(image_inbatch_embeddings,dim=0).unsqueeze(0))
            # import pdb; pdb.set_trace()
            batch['text_embeddings'] = torch.cat(text_embeddings, dim=0)
            batch['image_embeddings'] = torch.cat(image_embeddings,dim=0)
            
            batch["image_embeddings_ref"] = batch['image_embeddings'][:, 0, :] #image_after_features
            # import pdb; pdb.set_trace()
            # batch['text_embeddings'] = torch.cat((text_before_features.unsqueeze(1),torch.zeros((text_before_features.shape[0], 9, 768)).cuda()),dim=1)
            # batch['image_embeddings'] = torch.cat((image_after_features.unsqueeze(1),torch.zeros((image_after_features.shape[0], 9, 768)).cuda()),dim=1)
            # batch['image'] = batch['image'].permute(0,3,1,2)
            batch['ref'] = batch['ref'].float()
            batch['layout'] = batch['layout'].float()

            
        # import pdb; pdb.set_trace()
        z = self.autoencoder.encode( batch["image"] )
        # z_layout = self.autoencoder.encode(batch["layout"].permute(0,3,1,2))
        # inpainting_mask = draw_masks_from_boxes(batch['box_ref'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda() # batch['boxes']
        # masked_z = z * inpainting_mask   # 除 ref_img的周边区域
        # ref_z = z * (1-inpainting_mask)  # ref_img区域

        
        # ref = self.autoencoder.encode(batch["ref"])

        context_txt = self.text_encoder.encode( batch["caption"]  )
        ref_image_emb = self.image_encoder.encode( batch["ref"]  )  #做reshape就行,projection处1024改768
        context_img = ref_image_emb
        context = torch.cat((context_txt, ref_image_emb),dim=1)
        ref_image_emb_global = self.image_encoder_global.encode( batch["ref"]  )  #做reshape就行,projection处1024改768
        batch['ref_image_emb_global'] = ref_image_emb_global.float()

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()  # e.p: [306, 547]
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda() # batch['boxes']
            masked_z = z*inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
        
        box_layout = batch["layout"].permute(0,3,1,2)
        
        return z, box_layout, t, context, inpainting_extra_input, grounding_extra_input  #后2个都是空


    # def run_one_step(self, batch):
    #     x_start, x_start_masked, x_start_ref, t, context, context_txt, context_img, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
    #     noise = torch.randn_like(x_start)
    #     x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    #     x_noisy_masked = self.diffusion.q_sample(x_start=x_start_masked, t=t, noise=noise)
    #     x_noisy_ref = self.diffusion.q_sample(x_start=x_start_ref, t=t, noise=noise)
    #     # import pdb; pdb.set_trace()
    #     grounding_input = self.grounding_tokenizer_input.prepare(batch)  # keys: 'boxes', 'masks', 'text_masks', 'image_masks', 'text_embeddings', 'image_embeddings'
    #     input = dict(x=x_noisy_masked, 
    #                 timesteps=t, 
    #                 context=context_txt, 
    #                 inpainting_extra_input=inpainting_extra_input,
    #                 grounding_extra_input=grounding_extra_input,
    #                 grounding_input=grounding_input)
    #     model_output = self.model(input)
    #     # import pdb; pdb.set_trace()
    #     loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight  #[4,4,64,64]
    #     # 加 针对ref image的 loss
    #     input = dict(x=x_noisy_ref, 
    #                 timesteps=t, 
    #                 context=context_img, 
    #                 inpainting_extra_input=inpainting_extra_input,
    #                 grounding_extra_input=grounding_extra_input,
    #                 grounding_input=grounding_input)
    #     model_output_ref = self.model(input)         
    #     self.coefficient = 0.5
    #     loss_ref = torch.nn.functional.mse_loss(model_output_ref, noise) * self.coefficient
    #     # import pdb; pdb.set_trace()
    #     loss = loss + loss_ref
    #     self.loss_dict = {"loss": loss.item()}

    #     return loss 
    #  应该是对embedding做 mask?
    def run_one_step(self, batch):
        x_start, box_layout, t, context,  inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        # import pdb; pdb.set_trace()
        grounding_input = self.grounding_tokenizer_input.prepare(batch)  # keys: 'boxes', 'masks', 'text_masks', 'image_masks', 'text_embeddings', 'image_embeddings'
        input = dict(x=x_noisy,
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input,
                    layout = box_layout)
        model_output = self.model(input)
        
        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss 


    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 ) #+100000
        self.model.train()
        # import pdb; pdb.set_trace()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training  20001起始
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            loss.backward()
            self.opt.step() 
            self.scheduler.step()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)

            # import pdb; pdb.set_trace()
            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()

        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    
    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_caption=True):
    
        if out is None:
            out = self[index]

        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
        canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        W, H = img.size

        if print_caption:
            caption = out["caption"]
            print(caption)
            print(" ")

        boxes = []
        for box in out["boxes"]:    
            x0,y0,x1,y1 = box
            boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )
        img = draw_box(img, boxes)
        
        if return_tensor:
            return  torchvision.transforms.functional.to_tensor(img)
        else:
            img.save(name) 

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            # import pdb; pdb.set_trace()
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)
            # import pdb; pdb.set_trace()
            if self.config.use_mvimgnet:
                real_images_with_box_drawing = [] # we save this durining training for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                    # im = self.dataset_train.vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    im = self.vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)
                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)

                # real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
            else:
                # import pdb; pdb.set_trace()
                if "boxes" in batch:
                    real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                    for i in range(batch_here):
                        temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                        im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                        real_images_with_box_drawing.append(im)
                    real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
                else:
                    # keypoint case 
                    real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
                
            # TODO: 在这里加dino encoder
            # import pdb; pdb.set_trace()
            uc = self.text_encoder.encode( batch_here*[""] )
            uc_image = self.image_encoder.encode(batch_here*[ torch.zeros((1,3,224,224)) ])
            uc = torch.cat((uc,uc_image), dim=1)
            uc_image_global = self.image_encoder_global.encode(batch_here*[ torch.zeros((1,3,224,224)) ])
            context = self.text_encoder.encode(  batch["caption"]  )
            ref_image_emb = self.image_encoder.encode( batch["ref"].float()  )  #做reshape就行,projection处1024改768
            context = torch.cat((context, ref_image_emb),dim=1)
            ref_image_emb_global = self.image_encoder_global.encode( batch["ref"].float()  ) 

            batch['ref_image_emb_global'] = ref_image_emb_global.float()
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            # extra input for inpainting 
            inpainting_extra_input = None
            if self.config.inpaint_mode:
                z = self.autoencoder.encode( batch["image"] )
                inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
                masked_z = z*inpainting_mask
                inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
            
            grounding_extra_input = None
            # import pdb; pdb.set_trace()
            ####### use this when mvimgnet
            text_embeddings = []
            image_embeddings = []
            # list的维度其实是[]
            for tags in batch['positive_all']:
                text_inbatch_embeddings = []
                image_inbatch_embeddings = []
                tag_list = tags.split(',')
                for tag in tag_list:
                    inputs = self.processor(text=tag,return_tensors="pt", padding=True)  # batch['positive']是对应的object caption
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['pixel_values'] = batch['ref_processed'].cuda() # we use our own preprocessing without center_crop 
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    outputs = self.clip_model(**inputs)     
                    text_before_features = outputs.text_model_output.pooler_output # before projection feature
                    image_after_features = outputs.image_embeds

                    text_inbatch_embeddings.append(text_before_features)
                    image_inbatch_embeddings.append(image_after_features)
                text_embeddings.append(torch.cat(text_inbatch_embeddings,dim=0).unsqueeze(0))
                image_embeddings.append(torch.cat(image_inbatch_embeddings,dim=0).unsqueeze(0))
            # import pdb; pdb.set_trace()
            batch['text_embeddings'] = torch.cat(text_embeddings, dim=0)
            batch['image_embeddings'] = torch.cat(image_embeddings,dim=0)
            
            batch["image_embeddings_ref"] = batch['image_embeddings'][:, 0, :] #image_after_features

            # inputs = self.processor(text=batch['positive'],  return_tensors="pt", padding=True)  # batch['positive']是对应的object caption
            # inputs['input_ids'] = inputs['input_ids'].cuda()
            # inputs['pixel_values'] = batch['ref_processed'].cuda() # we use our own preprocessing without center_crop 
            # inputs['attention_mask'] = inputs['attention_mask'].cuda()
            # outputs = self.clip_model(**inputs)     

            # text_before_features = outputs.text_model_output.pooler_output # before projection feature
            # image_after_features = outputs.image_embeds # normalized after projection feature (CLIP aligned space)
            
            # batch["image_embeddings_ref"] = image_after_features
        
            # batch['text_embeddings'] = torch.cat((text_before_features.unsqueeze(1),torch.zeros((text_before_features.shape[0], 9, 768)).cuda()),dim=1)
            # batch['image_embeddings'] = torch.cat((image_after_features.unsqueeze(1),torch.zeros((image_after_features.shape[0], 9, 768)).cuda()),dim=1)
            # batch['image'] = batch['image']#.permute(0,3,1,2)
            batch['ref'] = batch['ref'].float()


            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            # TODO 这里需要把新推理出来的embeddings加进来
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            
            box_layout = batch["layout"].permute(0,3,1,2).float().cuda()
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          inpainting_extra_input=inpainting_extra_input,
                          grounding_extra_input=grounding_extra_input,
                          grounding_input=grounding_input,
                          layout = box_layout)
            # import pdb; pdb.set_trace()
            # print(input.keys())
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=3)  #原为5
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            masked_real_image =  batch["image"]*torch.nn.functional.interpolate(inpainting_mask, size=(512, 512)) if self.config.inpaint_mode else None
            self.image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, batch['ref'], batch["caption"], iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = self.text_encoder.state_dict(),
                    image_encoder = self.image_encoder.state_dict(),
                    image_encoder_global = self.image_encoder_global.state_dict(),
                    autoencoder = self.autoencoder.state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )


