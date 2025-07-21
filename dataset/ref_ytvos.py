"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
from dataset import transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random
from torchvision import transforms as torchvision_transforms
# from datasets.categories import ytvos_category_dict as category_dict
from dataset.mvimgnet import BaseDataset_t2i
from PIL import Image
from .data_utils1 import * 
import albumentations as A

category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}

ytvos_category_list = [
    'airplane', 'ape', 'bear', 'bike', 'bird', 'boat', 'bucket', 'bus', 'camel', 'cat', 'cow', 'crocodile', 
    'deer', 'dog', 'dolphin', 'duck', 'eagle', 'earless_seal', 'elephant', 'fish', 'fox', 'frisbee', 'frog', 
    'giant_panda', 'giraffe', 'hand', 'hat', 'hedgehog', 'horse', 'knife', 'leopard', 'lion', 'lizard', 
    'monkey', 'motorbike', 'mouse', 'others', 'owl', 'paddle', 'parachute', 'parrot', 'penguin', 'person', 
    'plant', 'rabbit', 'raccoon', 'sedan', 'shark', 'sheep', 'sign', 'skateboard', 'snail', 'snake', 'snowboard', 
    'squirrel', 'surfboard', 'tennis_racket', 'tiger', 'toilet', 'train', 'truck', 'turtle', 'umbrella', 'whale', 'zebra'
]

def convert_np_to_tensor(data):
    if isinstance(data, dict):
        return {k: convert_np_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_to_tensor(item) for item in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return data

def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.2: # else keep both features 0.5 改0.1 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.1)*1    # drop rate 0.5 改0.1
        text_masks = masks

    return image_masks, text_masks


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class BaseDataset_t2i(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    
    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            # A.HorizontalFlip(p=0.5),  #5.30 不做flip了
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


    def __getitem__(self, idx):
        while(True):
            try:
                idx = np.random.randint(0, len(self.data)-1)
                item = self.get_sample(idx)
                return item
            except:
                return None
                # idx = np.random.randint(0, len(self.data)-1)
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 
    
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
    
    # 对原图进行剪裁
    def center_crop(self, image, mask):
        height, width, _ = image.shape
        crop_width = width
        crop_height = width
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height

        cropped_image = image[start_y:end_y, :, :]
        cropped_mask = mask[start_y:end_y, :]

        return cropped_image, cropped_mask
    
    def center_crop_img(self, image):
        height, width, _ = image.shape
        crop_width = width
        crop_height = width
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height

        cropped_image = image[start_y:end_y, :, :]

        return cropped_image

    def process_pairs_customized_mvimagenet(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        # assert mask_score(ref_mask) > 0.90
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.01, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑
        # 这里控制，如果ref object在裁剪区域之内就用crop，如果在之外就pad
        
        y1,y2,x1,x2 = ref_box_yyxx
        # print('ref_box_yyxx_before:',ref_box_yyxx)
        
        #第二次训练不用了，任意位置生成, 这里Optional
        H, W, _ = masked_ref_image.shape
        
        ref_box_yyxx_tar = get_bbox_from_mask(tar_mask)
        y_tar1, y_tar2, x_tar1, x_tar2 = ref_box_yyxx_tar
        if y1 < H/2 - W/2 or y2 > H/2 + W/2 or y_tar1 < H/2 - W/2 or y_tar2 > H/2 + W/2:  # 超出边界，进行pad
            pass
        else:    #不超出边界，进行crop
            masked_ref_image = self.center_crop_img(masked_ref_image)
            tar_image, tar_mask = self.center_crop(tar_image,tar_mask)
            ref_image, ref_mask = self.center_crop(ref_image,ref_mask)
            ref_box_yyxx = get_bbox_from_mask(ref_mask)
            y1,y2,x1,x2 = ref_box_yyxx
            # print('ref_box_yyxx_after:',ref_box_yyxx)
        
        
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]  #(70,216,3) 只有该物体，将其抠出来. masked_ref_image1.png
        ref_mask = ref_mask[y1:y2,x1:x2]

        ratio = np.random.randint(10, 15) / 10 
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 0, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) #masked_ref_image3.png

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
        ref_mask = ref_mask_3[:,:,0]

        # Augmenting reference image
        #masked_ref_image_aug = self.aug_data(masked_ref_image) 
        
        # Getting for high-freqency map  这个可以留也可以不留。暂且留着
        if self.mode == 'train':
            masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        else:
            masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 
        masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        # 6.6 不做expand
        # tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.0,1.2]) #1.1  1.3
        # assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        
        # Cropping around the target object 
        # 6.6 不做expand
        # tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0]) # ratio=[1.3, 3.0]   
        tar_box_yyxx_crop = tar_box_yyxx
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx

        layout = np.zeros((tar_box_yyxx_crop[1]-tar_box_yyxx_crop[0], tar_box_yyxx_crop[3]-tar_box_yyxx_crop[2], 3), dtype=np.float32)
        layout[y1:y2,x1:x2,:] = [1.0, 1.0, 1.0]
        layout = pad_to_square(layout, pad_value = 0, random = False)
        layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        try:
            white_pixels = np.where(layout == 1.0)
            # layout = torch.from_numpy(layout)
            # 获取白色区域的边界
            bbox_top = np.min(white_pixels[0])
            bbox_bottom = np.max(white_pixels[0])
            bbox_left = np.min(white_pixels[1])
            bbox_right = np.max(white_pixels[1])

            # 图像尺寸
            img_height, img_width = 512,512 #(512,512,3)

            # 转换为相对坐标
            ymin = bbox_top / img_height
            ymax = bbox_bottom / img_height
            xmin = bbox_left / img_width
            xmax = bbox_right / img_width

            # (x0, y0) = relative_top
            # (x1, y1) = relative_right
            bbox = np.array([xmin, ymin, xmax, ymax])
            boxes = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 
        except:
            bbox = np.array([0, 0, 0, 0])
            boxes = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 
         #/255 #-1.0
        
        # Prepairing collage image
        # ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        
        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        
        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        # masked_ref_image_aug = masked_ref_image_aug  / 127.5 -1.0
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        # collage = collage / 127.5 - 1.0         # [-1,1]之间
        # collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

        item = dict(
                ref=masked_ref_image_aug.copy(),  # masked_ref_image_aug.copy()（224,224,3）
                jpg=cropped_target_image.copy(),  # (512,512,3)
                # tar_box_yyxx_crop=np.array(tar_box_yyxx_crop), 
                layout = layout.copy(),
                boxes = boxes.copy()
                ) 
        return item
    
class YTVOSDataset(BaseDataset_t2i):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    def __init__(self, img_folder: Path, ann_file: Path, transforms,   #return_masks: bool, 
                 num_frames: int, max_skip: int):
        
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        # self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        self.max_skip = max_skip
        
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    
        self.max_boxes = 10
        self.preprocess = torchvision_transforms.Compose([
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])
        self.dynamic = 2
        self.random_drop_embedding = 'both'
        self.prob_use_caption = 0.9
        self.prob_use_ref = 0.9
        self.mode = 'train'
        
        
    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])
    
    def prepare_metas(self):
        # read object information
        img_folder = self.img_folder
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)
            
            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                # sample_indx = random.randint(1, 3)
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)
    
                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')
                # print(img_path)
                # create the target
                label = torch.tensor(category_id) 
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0) 
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)])
            }
            
            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target) 
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)
            
            
            # imgs里面任意取两帧就行
            # Sampling frames
            T = len(imgs)
            min_interval = 3
            start_frame_index = np.random.randint(low=0, high=T - min_interval)
            end_frame_index = start_frame_index + np.random.randint(min_interval,  T - start_frame_index )
            end_frame_index = min(end_frame_index, T - 1)
            
            if random.uniform(0, 1) > 0.5 :  # 设置0.5的概率互换
                tmp = end_frame_index
                end_frame_index = start_frame_index
                start_frame_index = tmp
            else:
                pass
            
            # items = self.process_pairs_customized(imgs[start_frame_index], masks[start_frame_index], imgs[end_frame_index], masks[end_frame_index])  # add layout    #(500,333,3)
            # items['layout_all'], items['layout'],items['boxes'] = self.process_boxes_openimage(image_tensor, seg_mask, image_tensor, seg_mask)  
            source_img = (imgs[start_frame_index].numpy().transpose(1,2,0)*255).astype(np.uint8)  #(h,w,3) [0,1]改
            target_img = (imgs[end_frame_index].numpy().transpose(1,2,0)*255).astype(np.uint8)   #(h,w,3) [0,1]改
            source_mask = (target['masks'][start_frame_index].numpy() > 0).astype(np.uint8)     #(h,w) [bool)改
            target_mask = (target['masks'][end_frame_index].numpy() > 0).astype(np.uint8)      ##(h,w) [bool)改
            # target_img = PIL.Image.fromarray(imgs[end_frame_index].numpy().astype(np.uint8).transpose(1,2,0))
            # source_mask = PIL.Image.fromarray(masks[start_frame_index].numpy().astype(np.uint8))
            # target_mask = PIL.Image.fromarray(masks[end_frame_index].numpy().astype(np.uint8))
            if target_mask.any() and source_mask.any() and instance_check == True:
                item_with_collage = self.process_pairs_customized_mvimagenet(source_img, source_mask, target_img, target_mask) #全部[1920,1080]
            else:
                item_with_collage = {}
                item_with_collage['ref'] = np.zeros((224,224,3), dtype=np.float32)
                item_with_collage['jpg'] = np.zeros((512,512,3), dtype=np.float32)
                item_with_collage['layout'] = np.zeros((512,512,3), dtype=np.float32)
                item_with_collage['boxes'] = np.zeros((10,4), dtype=np.float32)
                
            item_with_collage['txt'] = target['caption']
            sampled_time_steps = self.sample_timestep()
            
            item_with_collage['time_steps'] = sampled_time_steps
            
            ## TODO class
            
            item_with_collage['positive'] = category  # change to class info
            prompt_list = []
            prompt_list.append(category)
            prompt_list.extend([''] * (self.max_boxes - 1))
            item_with_collage['positive_all'] = ','.join(prompt_list)
            
            
            item_with_collage['layout_all'] = item_with_collage['layout']
            image_crop = self.preprocess(item_with_collage['ref']).float()
            item_with_collage['ref_processed'] = image_crop
            item_with_collage['ref'] = item_with_collage['ref'].transpose(2,0,1).astype(np.float32)

            # add random drop
            #除了第一个其他都是0？
            array = torch.zeros(self.max_boxes) #np.zeros(self.max_boxes,dtype=np.int32)
            array[0] = 1
            item_with_collage['masks'] = array #.reshape(30,1)
            
            if self.random_drop_embedding != 'none':
                image_masks, text_masks = mask_for_random_drop_text_or_image_feature(item_with_collage['masks'], self.random_drop_embedding)
            else:
                image_masks = item_with_collage['masks']
                text_masks = item_with_collage['masks']

            item_with_collage["text_masks"] = text_masks  # item_with_collage['txt']
            item_with_collage["image_masks"] = image_masks #item_with_collage['txt']
            if random.uniform(0, 1) < self.prob_use_caption:
                item_with_collage["caption"] = item_with_collage['txt']
            else:
                item_with_collage["caption"] = ""
            if random.uniform(0, 1) < self.prob_use_ref:
                pass
            else:
                item_with_collage["ref"] = np.zeros((3,224,224), dtype=np.float32)#.astype(np.float32)

            # item_with_collage["caption"] = item_with_collage['txt'] #= 'a'+ class_key + 'on the beach'
            item_with_collage['box_ref'] = item_with_collage['boxes'][0:1]  #.unsqueeze(1)
            item_with_collage["image"] = item_with_collage["jpg"].transpose(2,0,1) #.transpose(1,2,0)

            item_with_collage = convert_np_to_tensor(item_with_collage)
                

        # return imgs, target
        return item_with_collage




def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks, 
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset
