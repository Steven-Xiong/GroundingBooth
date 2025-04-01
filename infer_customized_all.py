## this is for lvis validation set

import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


# from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from dataset.dreambooth_specific_box_all import DreamBoothDataset
from dataset.mvimgnet import MVImageNet_Grounding
from dataset.lvis import LvisDataset

from dataset.concat_dataset import ConCatDataset 

device = "cuda"


import torch
from PIL import Image
from transformers import AutoImageProcessor,AutoProcessor, CLIPModel, AutoModel
import torch.nn as nn
from torchmetrics.functional.multimodal import clip_score
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
clipprocessor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
dinoprocessor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinomodel = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
from torchvision import transforms
from PIL import Image
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pytorch_fid import fid_score

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    
    image_encoder = instantiate_from_config(config['image_encoder']).to(device).eval()
    image_encoder_global = instantiate_from_config(config['image_encoder_global']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    # import pdb; pdb.set_trace()
    text_encoder.load_state_dict( saved_ckpt["text_encoder"],strict = False  )  #版本问题, 不确定是否有影响
    image_encoder.load_state_dict( saved_ckpt["image_encoder"]  )
    image_encoder_global.load_state_dict( saved_ckpt["image_encoder_global"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, image_encoder, image_encoder_global, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
# inference可以借鉴，比如同时指定前景和背景的位置
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hed(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    hed_edge = Image.open(meta['hed_image']).convert("RGB")
    hed_edge = crop_and_resize(hed_edge)
    hed_edge = ( pil_to_tensor(hed_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "hed_edge" : hed_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_canny(meta, batch=1):
    """ 
    The canny edge is very sensitive since I set a fixed canny hyperparamters; 
    Try to use the same setting to get edge 

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img,100,200)
    edges = PIL.Image.fromarray(edges)

    """
    
    pil_to_tensor = transforms.PILToTensor()

    canny_edge = Image.open(meta['canny_image']).convert("RGB")
    canny_edge = crop_and_resize(canny_edge)

    canny_edge = ( pil_to_tensor(canny_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "canny_edge" : canny_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_depth(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_normal(meta, batch=1):
    """
    We only train normal model on the DIODE dataset which only has a few scene.

    """
    
    pil_to_tensor = transforms.PILToTensor()

    normal = Image.open(meta['normal']).convert("RGB")
    normal = crop_and_resize(normal)
    normal = ( pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5

    out = {
        "normal" : normal.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 





def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass 
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def run(batch, caption, config, meta, starting_noise=None):

    
    context = text_encoder.encode(  [caption]*config.batch_size  )
    ref_image_emb = image_encoder.encode( batch["ref"][0]*config.batch_size  )  #做reshape就行,projection处1024改768
    context = torch.cat((context, ref_image_emb),dim=1)
    ref_image_emb_global = image_encoder_global.encode( batch["ref"][0]*config.batch_size  )
    batch['ref_image_emb_global'] = ref_image_emb_global.float() 

    uc = text_encoder.encode( config.batch_size*[""] ) #torch.zeros([config.batch_size, 77, 768]).cuda() 
    uc_image = image_encoder.encode(config.batch_size *[ torch.zeros((1,3,224,224)) ]) #torch.zeros([config.batch_size, 257, 768]).cuda() 
    # print('uc_image.shape', uc_image.shape)
    uc = torch.cat((uc,uc_image), dim=1)
    uc_image_global = image_encoder_global.encode(config.batch_size*[ torch.zeros((1,3,224,224)) ])
    if args.negative_prompt is not None:
        uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )


    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type = meta.get("alpha_type") ) # type=[1.0,0.0,0.0]
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).cuda()
        
        input_image = F.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
    

    # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)
    
    grounding_input['masks'] = grounding_input['masks'].to(device)
    grounding_input['boxes'] = grounding_input['boxes'].to(device)
    grounding_input['text_masks'] = grounding_input['text_masks'].to(device)
    grounding_input['image_masks'] = grounding_input['image_masks'].to(device)
    grounding_input['text_embeddings'] = grounding_input['text_embeddings'].to(device)
    grounding_input['image_embeddings'] = grounding_input['image_embeddings'].to(device)
    grounding_input['ref_embeddings'] = grounding_input['ref_embeddings'].to(device)
    grounding_input['ref_box'] = grounding_input['ref_box'].to(device)
    
    box_layout = batch["layout"].permute(0,3,1,2).float().cuda()
    # import pdb; pdb.set_trace()
    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,
                layout = box_layout
            )

    
    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    # import pdb; pdb.set_trace()

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)


    # - - - - - save - - - - - #
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    # print(image_ids)
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample1 = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample1.astype(np.uint8))
        sample.save(  os.path.join(output_folder, img_name)   )
        
    return sample1


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=5, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")   
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--dataset", type = str, default = 'tsv')  # 'tsv', 'mvimgnet', 'dreambench'
    
    #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    parser.add_argument("--ckpt_path", type = str)
    parser.add_argument("--position", type=float, nargs=4, default = (0.25,0.25,0.75,0.75))  #这里设定position
    args = parser.parse_args()
    
   

    meta_list = [ 
        
        # - - - - - - - - GLIGEN on text dino grounding for generation - - - - - - - - # 
        # 自己的text dino grounding
        dict(
            ckpt = "OUTPUT/COCO_text_dino_ground/tag00/checkpoint_latest.pth",
            prompt = "an alarm clock sitting on the beach",
            images = ['inference_images/clock.png'],
            phrases = ['alarm clock'],
            locations = [ [0.3,0.2,0.7,0.9] ],
            alpha_type = [1.0, 0.0, 0.0],
            save_folder_name="generation_box_image"
        ),

       


    ]
    
    tarting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
   

    config = OmegaConf.load('./configs/inference.yaml')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is:', device)
    time = args.folder.split('/')[-1] #'5.19_text_dino_mvimgnet'
    dir_path = os.path.join('OUTPUT_test',time)
    os.makedirs(dir_path,exist_ok=True)
    print('dir_path:',dir_path)
    
    save_dir_gen = 'OUTPUT_test/'+ time + '/gen'
    save_dir_gt = 'OUTPUT_test/'+ time +'/gt'
    save_dir_ref = 'OUTPUT_test/'+ time +'/ref'
    save_dir_layout = 'OUTPUT_test/'+ time + '/layout'
    save_dir_ref_layout = 'OUTPUT_test/'+ time + '/ref_layout'
    save_dir_all = 'OUTPUT_test/'+ time + '/all'
    caption_path = 'OUTPUT_test/'+ time +'/caption.txt'
    
    version = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)
        
    if not os.path.exists(save_dir_gen):
        # import pdb; pdb.set_trace()
        os.mkdir(save_dir_gen)
    if not os.path.exists(save_dir_gt):
        os.mkdir(save_dir_gt)    
    if not os.path.exists(save_dir_ref):
        os.mkdir(save_dir_ref)    
    if not os.path.exists(save_dir_layout):
        os.mkdir(save_dir_layout)   
    if not os.path.exists(save_dir_ref_layout):
        os.mkdir(save_dir_ref_layout)    
    if not os.path.exists(save_dir_all):
        os.mkdir(save_dir_all)    
    # import pdb; pdb.set_trace()
    # data = os.listdir( '/project/osprey/scratch/x.zhexiao/GLIGEN/data/dreambooth_withmask/dataset/')
    # print(data)
    if args.dataset == 'dreambench':
        dataset_test = DreamBoothDataset(fg_dir = '/project/osprey/scratch/x.zhexiao/GLIGEN/data/dreambooth_withmask_selected/dataset/', bg_dir = '/project/osprey/scratch/x.zhexiao/GLIGEN/data/dreambooth_withmask_selected/dataset',
        caption_dir = '/project/osprey/scratch/x.zhexiao/GLIGEN/data/dreambooth/prompts_and_classes.txt', position = args.position)  # 指定位置生成
    elif args.dataset == 'mvimgnet':
        dataset_test = MVImageNet_Grounding( txt='/project/osprey/scratch/x.zhexiao/GLIGEN/data/MVImgNet_full/list.txt', image_dir='/project/osprey/scratch/x.zhexiao/GLIGEN/data/MVImgNet_full/', mode = 'test') 
    elif args.dataset == 'lvis':
        dataset_test = dataset_train = LvisDataset(image_dir= 'data/lvis_v1/val2017', json_path='data/lvis_v1/lvis_v1_val.json', mode = 'test')
    else:
        dataset_test = ConCatDataset(config.test_dataset_names, 'DATA', train=False, repeats=None)
    
    # dataset_test.getitem(1)
    # dataset_test = ConCatDataset(config.test_dataset_names, 'DATA', train=False, repeats=None)
    dataloader = DataLoader(dataset_test, num_workers=4, batch_size=1, shuffle=False)
    
        
    CLIP_T = []
    CLIP_I = []
    DINO_I = []
    YOLO_SCORE = []
    pred_boxes = []
    gt_boxes = []
    from ultralytics import YOLO

    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    yolo_model = YOLO("./pretrained/yolov8n.pt")  # load a pretrained model (recommended for training)
    
    
    # - - - - - prepare models - - - - - # 
    # import pdb; pdb.set_trace()
    meta = meta_list[0]
    
    model, autoencoder, text_encoder, image_encoder, image_encoder_global, diffusion, config = load_ckpt(args.ckpt_path) #对应自己的模型
    print('load checkpoint form: ', args.ckpt_path)
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)

    def move_tensors_to_cuda(tensor_dict):
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 遍历字典中的所有键值对
            for key, tensor in tensor_dict.items():
                # 将Tensor移动到CUDA设备上
                tensor_dict[key] = tensor.cuda()
        else:
            print("CUDA is not available. Tensors were not moved.")
        return tensor_dict
        
    for i, batch in enumerate(dataloader):
        
        for j, prompt in enumerate(batch['caption']):
            
            ref_image = batch['ref'][0]
            gt_image = batch['image'][0]  
            caption = prompt[0]
            layout = batch['layout_all'][0]
            print(caption)
            
            ref_layout = batch['layout'][0]
            gt_box_ref = batch['box_ref'][0]
            object_name = batch['object_name'][0]
            
            
            text_embeddings = []
            image_embeddings = []
            for tags in batch['positive_all']:
                text_inbatch_embeddings = []
                image_inbatch_embeddings = []
                tag_list = tags.split(',')
                for tag in tag_list:
                    inputs = processor(text=tag,return_tensors="pt", padding=True)  # batch['positive']是对应的object caption
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['pixel_values'] = batch['ref_processed'].cuda() # we use our own preprocessing without center_crop 
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    outputs = clip_model(**inputs)     
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
            
            batch['ref'] = batch['ref'].float()
            # import pdb; pdb.set_trace()
            gen_image = run(batch,caption,config,meta)   #(ref_image, ref_mask, gt_image.copy(), tar_mask)
            
            # image_name = batch['img_path'][0].split('/')[-1]
            image_name = object_name + '_' + caption + '.png'
            gen_path = os.path.join(save_dir_gen, image_name)
            gt_path = os.path.join(save_dir_gt, image_name)
            ref_path = os.path.join(save_dir_ref,image_name)
            layout_path = os.path.join(save_dir_layout,image_name)
            ref_layout_path = os.path.join(save_dir_ref_layout,image_name)
            concat_path = os.path.join(save_dir_all,image_name)
            # import pdb; pdb.set_trace()
            gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
            ref_image = cv2.cvtColor((ref_image.numpy()* 255.0).transpose(1,2,0), cv2.COLOR_BGR2RGB)
            gt_image = cv2.cvtColor(((gt_image.numpy()+1.0)/2 * 255.0).transpose(1,2,0), cv2.COLOR_BGR2RGB)
            layout_image = cv2.cvtColor((layout.numpy()* 255.0), cv2.COLOR_BGR2RGB)
            ref_layout_image = cv2.cvtColor((ref_layout.numpy()* 255.0), cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(gen_path,gen_image)
            cv2.imwrite(gt_path,gt_image)
            cv2.imwrite(ref_path,ref_image)
            cv2.imwrite(layout_path,layout_image)
            cv2.imwrite(ref_layout_path,ref_layout_image)
            try:
                results_concat = np.concatenate((gen_image, gt_image), axis=0)
                resized_ref = Image.fromarray(ref_image.astype(np.uint8))
                resized_ref_image = np.array(resized_ref.resize((512, 512), Image.ANTIALIAS))
                results_concat = np.concatenate((results_concat, Image.fromarray(resized_ref_image)), axis=0)
                results_concat = np.concatenate((results_concat, layout_image), axis=0)
                results_concat = np.concatenate((results_concat, ref_layout_image), axis=0)
                cv2.imwrite(concat_path,results_concat)
            except:
                pass
        

