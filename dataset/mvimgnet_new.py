import torch 
# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
# from ldm.modules.encoders.modules import BERTEmbedder
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import os 
import math
import clip 
from PIL import Image
from torchvision import transforms
import multiprocessing
from zipfile import ZipFile 
from io import BytesIO

from PIL import Image
# from lang_sam import LangSAM
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from pycocotools.coco import COCO
from pycocotools.mask import decode
from torch.utils.data import Dataset
import numpy as np
from lvis import LVIS
import csv
import glob
from .data_utils1 import * 

import json
import cv2
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
import matplotlib.pyplot as plt
import random
import torchvision

def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_annotations(annotations):
    for anno in annotations:
        anno.pop("segmentation", None)
        anno.pop("area", None)
        anno.pop("iscrowd", None)


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 
    
    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)


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



class2id = {'bag': 0, 'bottle': 1, 'wahser': 2, 'vessel': 3, 'train': 4, 'telephone': 5, 'table': 6, 'stove': 7, 'sofa': 8, 'skateboard': 9, 'rifle': 10, 'pistol': 11, 'remote control': 12, 'printer': 13, 'flowerpot': 14, 'pillow': 15, 'piano': 16, 'mug': 17, 'motorcycle': 18, 'microwave': 19, 'microphone': 20, 'mailbox': 21, 'loudspeaker': 22, 'laptop': 23, 'lamp': 24, 'knife': 25, 'pot': 26, 'helmet': 27, 'guitar': 28, 'bookshelf': 29, 'faucet': 30, 'earphone': 31, 'display': 32, 'dishwasher': 33, 'computer keyboard': 34, 'clock': 35, 'chair': 36, 'car': 37, 'cap': 38, 'can': 39, 'camera': 40, 'caninet': 41, 'bus': 42, 'bowl': 43, 'bicycle': 44, 'bench': 45, 'bed': 46, 'bathtub': 47, 'basket': 48, 'ashcan': 49, 'airplane': 50, 'umbrella': 51, 'plush toy': 52, 'toy figure': 53, 'towel': 54, 'toothbrush': 55, 'toy bear': 56, 'toy cat': 57, 'toy bird': 58, 'toy insect': 59, 'toy cow': 60, 'toy dog': 61, 'toy monkey': 62, 'toy elephant': 63, 'toy fish': 64, 'toy horse': 65, 'toy sheep': 66, 'toy mouse': 67, 'toy tiger': 68, 'toy rabbit': 69, 'toy dragon': 70, 'toy snake': 71, 'toy chook': 72, 'toy pig': 73, 'rice cooker': 74, 'pressure cooker': 75, 'toaster': 76, 'dryer': 77, 'battery': 78, 'curtain': 79, 'blackboard eraser': 82, 'bucket': 83, 'calculator': 85, 'candle': 86, 'cassette': 87, 'cup sleeve': 88, 'computer mouse': 90, 'easel': 93, 'fan': 94, 'cookie': 96, 'fries': 97, 'donut': 98, 'coat rack': 99, 'guitar stand': 100, 'can opener': 101, 'flashlight': 102, 'hammer': 103, 'scissors': 104, 'screw driver': 105, 'spanner': 106, 'hanger': 107, 'jug': 108, 'fork': 109, 'chopsticks': 110, 'spoon': 111, 'ladder': 112, 'ceiling lamp': 113, 'wall lamp': 114, 'lamp post': 115, 'light switch': 116, 'mirror': 118, 'paper box': 119, 'wheelchair': 120, 'walking stick': 121, 'picture frame': 122, 'shower': 124, 'toilet': 125, 'sink': 126, 'power socket': 127, 'Bagged snacks': 129, 'Tripod': 130, 'Selfie stick': 131, 'Hair dryer': 132, 'Lipstick': 133, 'Glasses': 134, 'Sanitary napkin': 135, 'Toilet paper': 136, 'Rockery': 137, 'Chinese hot dishes': 138, 'Root carving': 139, 'Flower': 141, 'Book': 144, 'Pipe PVC Metal pipe': 145, 'Projector': 146, 'Cabinet Air Conditioner': 147, 'Desk Air Conditioner': 148, 'Refrigerator': 149, 'Percussion': 150, 'Strings': 152, 'Wind instruments': 153, 'Balloons': 154, 'Scarf': 155, 'Shoe': 156, 'Skirt': 157, 'Pants': 158, 'Clothing': 159, 'Box': 160, 'Soccer': 161, 'Roast Duck': 162, 'Pizza': 163, 'Ginger': 164, 'Cauliflower': 165, 'Broccoli': 166, 'Cabbage': 167, 'Eggplant': 168, 'Pumpkin': 169, 'winter melon': 170, 'Tomato': 171, 'Corn': 172, 'Sunflower': 173, 'Potato': 174, 'Sweet potato': 175, 'Chinese cabbage': 176, 'Onion': 177, 'Momordica charantia': 178, 'Chili': 179, 'Cucumber': 180, 'Grapefruit': 181, 'Jackfruit': 182, 'Star fruit': 183, 'Avocado': 184, 'Shakyamuni': 185, 'Coconut': 186, 'Pineapple': 187, 'Kiwi': 188, 'Pomegranate': 189, 'Pawpaw': 190, 'Watermelon': 191, 'Apple': 192, 'Banana': 193, 'Pear': 194, 'Cantaloupe': 195, 'Durian': 196, 'Persimmon': 197, 'Grape': 198, 'Peach': 199, 'power strip': 200, 'Racket': 202, 'Toy butterfly': 203, 'Toy duck': 204, 'Toy turtle': 205, 'Bath sponge': 206, 'Glove': 207, 'Badminton': 208, 'Lantern': 209, 'Chestnut': 211, 'Accessory': 212, 'Shovel': 214, 'Cigarette': 215, 'Stapler': 216, 'Lighter': 217, 'Bread': 218, 'Key': 219, 'Toothpaste': 220, 'Swin ring': 221, 'Watch': 222, 'Telescope': 223, 'Eggs': 224, 'Bun': 225, 'Guava': 226, 'Okra': 227, 'Tangerine': 228, 'Lotus root': 229, 'Taro': 230, 'Lemon': 231, 'Garlic': 232, 'Mango': 233, 'Sausage': 234, 'Besom': 235, 'Lock': 237, 'Ashtray': 238, 'Conch': 240, 'Seafood': 241, 'Hairbrush': 243, 'Ice cream': 244, 'Razor': 245, 'Adhesive hook': 246, 'Hand Warmer': 247, 'Thermometer': 250, 'Bell': 251, 'Sugarcane': 252, 'Adapter(Water pipe)': 253, 'Calendar': 254, 'Insecticide': 261, 'Electric saw': 263, 'Inflator': 265, 'Ironmongery': 266, 'Bulb': 267}

id2class = {v:k for k,v in class2id.items()}

from PIL import Image, ImageDraw

def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline =colors[bid % len(colors)], width=4)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


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
                idx = np.random.randint(0, len(self.data)-1)
                
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
    
    def draw_bboxes_all(self, tensor, boxes, color=(1.0, 1.0, 1.0), thickness=2):
        """
        在PyTorch图像张量上绘制边界框边框，针对[H, W, C]的张量布局。

        参数:
        - tensor: 图像张量，形状为(H, W, C)，H为高度，W为宽度，C为通道数。
        - boxes: 边界框列表，每个边界框的格式为[x_min, y_min, x_max, y_max]，坐标为归一化值。
        - color: 边框的颜色，默认为白色。
        - thickness: 边框的厚度。
        """
        H, W, C = tensor.shape
        # import pdb; pdb.set_trace()
        
        x_min, y_min, x_max, y_max = boxes
        x_min, y_min, x_max, y_max = int(x_min * W), int(y_min * H), int(x_max * W), int(y_max * H)

        # 绘制顶部和底部
        tensor[y_min:y_min+thickness, x_min:x_max, :] = torch.tensor(color)
        tensor[y_max-thickness:y_max, x_min:x_max, :] = torch.tensor(color)

        # 绘制左侧和右侧
        tensor[y_min:y_max, x_min:x_min+thickness, :] = torch.tensor(color)
        tensor[y_min:y_max, x_max-thickness:x_max, :] = torch.tensor(color)

        return tensor

    def process_pairs_customized_mvimagenet(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        assert mask_score(ref_mask) > 0.90
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        # import pdb; pdb.set_trace()
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑
        # 这里控制，如果ref object在裁剪区域之内就用crop，如果在之外就pad
        
        y1,y2,x1,x2 = ref_box_yyxx
        # print('ref_box_yyxx_before:',ref_box_yyxx)
        
        ''' #第二次训练不用了，任意位置生成
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
        '''
        ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
        # import pdb; pdb.set_trace()
        # tar_box_yyxx_crop = box2squre(tar_image, ref_box_yyxx)
        ref_image = pad_to_square(ref_image, pad_value = 0)
        # tar_box_yyxx_squared = box2squre(ref_image, ref_box_yyxx) # square之后的bbox坐标
        
        # ratio = 512 / ref_image.shape[0]
        # y1_new,y2_new,x1_new,x2_new = tar_box_yyxx_squared 
        # y1_new,y2_new,x1_new,x2_new = int(y1_new * ratio), int(y2_new * ratio),int(x1_new*ratio),int(x2_new * ratio)

        jpg = (cv2.resize(ref_image.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)/ 127.5 - 1.0).astype(np.float32) # [-1,1]之间
        # [-1,1]之间
        
        ###################### 整图的bbox
        # ref_image = pad_to_square(ref_image, pad_value = 0)
        ref_mask_3_squared = pad_to_square(ref_mask_3, pad_value = 0)
        ref_mask_3_squared = cv2.resize(ref_mask_3_squared, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref_mask_squared =  ref_mask_3_squared[:,:,0]
        y1_new, y2_new, x1_new, x2_new = get_bbox_from_mask(ref_mask_squared)
        
        layout = np.zeros((512,512,3), dtype=np.float32)
        layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 1.0, 1.0]
        # layout = pad_to_square(layout, pad_value = 0, random = False)
        layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

        bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
        boxes = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 

        layout_multi = torch.zeros(512, 512, 3)
        layout_multi = self.draw_bboxes_all(layout_multi, bbox).cpu().numpy()  #只有一个box就不循环了

        ############################

        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]  #(70,216,3) 只有该物体，将其抠出来. masked_ref_image1.png
        ref_mask = ref_mask[y1:y2,x1:x2]

        # ratio = np.random.randint(11, 15) / 10 
        # masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
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
        # import pdb; pdb.set_trace()
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        # 6.6 不做expand
        # tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.0,1.1]) #1.1  1.3
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        
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

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        # import pdb; pdb.set_trace()
        collage = np.zeros(cropped_target_image.shape, dtype=np.uint8)  #改成ones？
        # collage = cropped_target_image.copy()   #这里collage是否改为不要背景？ 

        collage[y1:y2,x1:x2,:] = ref_image_collage
        
        # import pdb; pdb.set_trace()
        collage_mask = cropped_target_image.copy() * 0.0   #这里翻一下, mask掉背景? 应该不用改
        collage_mask[y1:y2,x1:x2,:] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]
        # import pdb; pdb.set_trace()
        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]
        # import pdb; pdb.set_trace()
        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32) #可以直接当做layout
        collage_mask[collage_mask == 2] = -1
        # import pdb; pdb.set_trace()
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        # masked_ref_image_aug = masked_ref_image_aug  / 127.5 -1.0
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0         # [-1,1]之间
        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
        

        item = dict(
                ref=masked_ref_image_aug.copy(),  # masked_ref_image_aug.copy()（224,224,3）
                jpg=jpg.copy(),  # (512,512,3)
                # tar_box_yyxx_crop=np.array(tar_box_yyxx_crop), 
                layout = layout.copy(),
                boxes = boxes.copy(),
                layout_all = layout_multi.copy()
                ) 
        return item



class MVImageNet_Grounding(BaseDataset_t2i):
    def __init__(self, txt, image_dir, mode ='train'):
        # import pdb; pdb.set_trace()
        with open(txt,"r") as f:
            data = f.read().split('\n')[:-1]    
        self.image_dir = image_dir 
        self.data = data
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.mask_dir = image_dir + 'mask/'
        self.caption_dir = image_dir + 'captions/all.csv'
        # import pdb; pdb.set_trace()
        # load captions
        self.caption_index = {}
        with open(self.caption_dir, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                path, caption = os.path.join(row[0].split('/')[-4],row[0].split('/')[-3],row[0].split('/')[-2],row[0].split('/')[-1]), row[1]
                self.caption_index[path] = caption
        self.mode = mode
        if self.mode == 'train':
            self.random_drop_embedding = 'both'
            self.prob_use_caption = 0.9
            self.prob_use_ref = 0.9
        else:
            self.random_drop_embedding = 'none'
            self.prob_use_caption = 1.0
            self.prob_use_ref = 1.0
        # image_count = 0
        # for folder in self.data:
        #     image_count += len(glob.glob(folder.replace('MVDir/', self.image_dir)))
        # import pdb; pdb.set_trace()
        
        # version = "openai/clip-vit-large-patch14"
        # self.clip_model = CLIPModel.from_pretrained(version) #.cuda()
        # self.processor = CLIPProcessor.from_pretrained(version)
        # self.transform_to_pil = transforms.ToPILImage()
        # # import pdb; pdb.set_trace()
        # self.get_clip_feature = get_clip_feature(model=self.clip_model, processor=self.processor,input=None, is_image=True)
        # self.projection_matrix = torch.load('projection_matrix') #.cuda()
        self.max_boxes = 10
        
        # import pdb; pdb.set_trace()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])
        
    def __len__(self):
        image_count = 0
        for folder in self.data:
            image_count +=   len(glob.glob( os.path.join(self.image_dir, folder,'images'))) #len(glob.glob(folder.replace('MVDir/', self.image_dir)))
        return image_count

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    def get_alpha_mask(self, mask_path):
        # import pdb; pdb.set_trace()
        image = cv2.imread( mask_path) #, cv2.IMREAD_UNCHANGED
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask
    
    def seg2bbox(self, mask, ratio=0.1):  #borrow from instantbooth
        mask = mask[:, :, 0]
        h, w= mask.shape[0], mask.shape[1]
        # crop exact bbox from mask. 
        y, x = np.where(mask!=0)
        xmin, xmax = np.min(x), np.max(x)+1
        ymin, ymax = np.min(y), np.max(y)+1
        tight_crop = (ymin, xmin, ymax, xmax)

        # expand according to ratio.
        ybox, xbox = ymax - ymin, xmax - xmin
        ycenter, xcenter = (ymin + ymax) // 2, (xmin + xmax) // 2
        ratio = min(ratio, h*1./ybox - 1., w*1./xbox - 1.)
        ynew, xnew = int(ybox * (1+ratio)), int(xbox * (1+ratio))
        ymin_new, ymax_new = max(ycenter - ynew//2, 0), min(ycenter + ynew//2, h)
        xmin_new, xmax_new = max(xcenter - xnew//2, 0), min(xcenter + xnew//2, w)
        expanded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)
        
        # pad to square bbox.
        ybox_new, xbox_new = ymax_new-ymin_new, xmax_new - xmin_new
        if xbox_new < ybox_new:
            pad = (ybox_new )//2
            xmin_new, xmax_new = max(xcenter - pad, 0), min(xcenter + pad, w)
        else:
            pad = (xbox_new )//2
            ymin_new, ymax_new = max(ycenter - pad, 0), min(ycenter + pad, h)
        # cropped_mask_pad = mask[ymin_new:ymax_new, xmin_new:xmax_new]
        # cv2.imwrite('cropped_mask_pad.png', cropped_mask_pad)
        padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)

        return padded_crop
    
   

    def get_sample(self, idx):
        
        # object_dir = self.data[idx].replace('MVDir/', self.image_dir)
        # 手动加mask dir
        # mask_dir = self.data[idx].replace('MVDir/', self.mask_dir).replace('/images','')
        # import pdb; pdb.set_trace()
        mask_dir = os.path.join(self.mask_dir, self.data[idx])
        object_dir = os.path.join(self.image_dir, self.data[idx],'images')

    
        frames = os.listdir(object_dir)
        frames = [ i for i in frames ] #if '.png' in i

        # Sampling frames
        min_interval = len(frames)  // 8
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        end_frame_index = min(end_frame_index, len(frames) - 1)
        # 5.30 用一种view change比较小的训练方式：
        # min_interval = len(frames)  // 8
        # start_frame_index = np.random.randint(low=min_interval, high=len(frames) - min_interval)
        # end_frame_index = start_frame_index + np.random.randint(-min_interval,  min_interval)
        # end_frame_index = min(end_frame_index, len(frames) - 1)
        # end_frame_index = max(end_frame_index, 0)

        # Get image path
        ref_mask_name = frames[start_frame_index] + '.png'
        tar_mask_name = frames[end_frame_index] + '.png'  #最后一帧做target

        ref_image_name = frames[start_frame_index] # ref_mask_name.split('_')[0] #+ '.jpg'
        tar_image_name = frames[end_frame_index] # tar_mask_name.split('_')[0] #+ '.jpg'

        ref_mask_path = os.path.join(mask_dir, ref_mask_name)
        tar_mask_path = os.path.join(mask_dir, tar_mask_name)
        ref_image_path = os.path.join(object_dir, ref_image_name)
        tar_image_path = os.path.join(object_dir, tar_image_name) 

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path).astype(np.uint8)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        # # add resize
        # ref_image = cv2.resize(ref_image, (512,512))

        tar_image = cv2.imread(tar_image_path).astype(np.uint8)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()
        if ref_mask_path.endswith('bg_removed.png.png') == False:  #
            ref_mask = self.get_alpha_mask(ref_mask_path)
            tar_mask = self.get_alpha_mask(tar_mask_path)
        else:
            ref_mask = np.full((tar_image.shape[0], tar_image.shape[1]), 255, dtype=np.uint8)
            tar_mask = np.full((tar_image.shape[0], tar_image.shape[1]), 255, dtype=np.uint8)
        
        # import pdb; pdb.set_trace()
        target_index = os.path.join(tar_image_path.split('/')[-4],tar_image_path.split('/')[-3],tar_image_path.split('/')[-2],tar_image_path.split('/')[-1])
        caption = self.caption_index[target_index]
        # import pdb; pdb.set_trace()
        bbox = self.seg2bbox(np.stack([tar_mask,tar_mask,tar_mask],-1))  #将seg map补全成mask,对应1920*1080
        # 创建一个全黑的图片
        
        # print('bbox:',bbox)
        # # 已知的bbox坐标：(ymin, xmin, ymax, xmax)
        # padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)  # 请替换为实际的坐标值
        # 在bbox区域内填充为白色（或其他颜色）
        # 注意：颜色设置为(255, 255, 255)是白色，(R, G, B)格式

        # layout = np.zeros((tar_image.shape[0], tar_image.shape[1], 3), dtype=np.uint8)
        # layout[bbox[0]:bbox[2], bbox[1]:bbox[3]] = [255, 255, 255]


        item_with_collage = self.process_pairs_customized_mvimagenet(ref_image, ref_mask, tar_image, tar_mask) #全部[1920,1080]      
        # import pdb; pdb.set_trace()
        # ref_image_resized = pad_to_square(ref_image, pad_value = 255, random = False)
        # ref_image_resized = cv2.resize(ref_image_resized.astype(np.uint8), (224,224)).astype(np.uint8) / 255
        # plt.imsave('ref_img_resized.jpg',item_with_collage['ref'])
        # item_with_collage['ref'] = ref_image_resized  #这样不对。应该剪裁居中并抠图，放进han'hu
        item_with_collage['anno_id'] = target_index   # target image path
        item_with_collage['txt'] = caption
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        
    
        # import pdb; pdb.set_trace()
        #找对应tag
        folder = object_dir.split('/')[-3]
        id2class = {v: k for k, v in class2id.items()}
        folder_name = folder
        class_key = id2class[int(folder_name)]

        item_with_collage['positive'] = class_key
        
        prompt_list = []
        prompt_list.append(class_key)
        prompt_list.extend([''] * (self.max_boxes - 1))
        item_with_collage['positive_all'] = ','.join(prompt_list)
        # import pdb; pdb.set_trace()
            
        # bbox需要归一化,是吗
        # bbox = np.array(bbox,dtype=np.float32)   #这里的bbox对应的是原图尺寸 1920*1080
        # bbox[0],bbox[1],bbox[2], bbox[3] = bbox[0]/tar_image.shape[0], bbox[1]/tar_image.shape[1], bbox[2]/tar_image.shape[0],bbox[3]/tar_image.shape[1]
        
        # y0, x0, y1, x1 = bbox[0],bbox[1],bbox[2], bbox[3]
        # bbox[0],bbox[1],bbox[2], bbox[3] = x0, y0, x1, y1
        # item_with_collage['boxes'] = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 
        
        # y1,y2,x1,x2 = item_with_collage['tar_box_yyxx_crop']
        # item_with_collage['layout_all'] = item_with_collage['layout']
        # import pdb; pdb.set_trace()
        image_crop = self.preprocess(item_with_collage['ref']).float()
        item_with_collage['ref_processed'] = image_crop
        # import pdb; pdb.set_trace()
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

        return item_with_collage          #{'positive': item_with_collage['positive'], 'anno_id': item_with_collage['anno_id'], 'image_crop': image_crop}
