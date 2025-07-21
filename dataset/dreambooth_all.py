import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
# from .data_utils import * 
# from .base import BaseDataset

from torch.utils.data import Dataset
import json
import cv2
import numpy as np
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
from torchvision import transforms
from .data_utils1 import * 

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
            if random.uniform(0, 1) < 0.2: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.1)*1
        text_masks = masks

    return image_masks, text_masks

class2id = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal"
}

# class2prompt = {
#     "backpack":'a {} in the jungle',
#     "backpack_dog" : 'a {} in the snow',
#     "bear_plushie":'a {} on the beach',
#     "berry_bowl": 'a {} on a cobblestone street',
#     "can":'a {} on top of pink fabric',
#     "candle":'a {} on top of a wooden floor',
#     "clock":'a {} with a city in the background',
#     "colorful_sneaker":'a {} with a mountain in the background',
#     "duck_toy":'a {} with a blue house in the background',
#     "fancy_boot":'a {} on top of a purple rug in a forest',
#     "grey_sloth_plushie":'a {} with a wheat field in the background',
#     "monster_toy":'a {} with a tree and autumn leaves in the background',
#     "pink_sunglasses":'a {} with the Eiffel Tower in the background',
#     "poop_emoji":'a {} floating on top of water',
#     "rc_car":'a {} floating in an ocean of milk',
#     'a {} on top of green grass with sunflowers around it',
#     'a {} on top of a mirror',
#     'a {} on top of the sidewalk in a crowded street',
#     'a {} on top of a dirt road',
#     "red_cartoon":'a {} on top of a white rug',
#     "robot_toy":'a red {}',
#     "shiny_sneaker":'a purple {}',
#     "teapot":'a shiny {}',
#     "vase":'a wet {}',
#     "wolf_plushie":'a cube shaped {}'
# }

OBJECT = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal"
}

LIVE_OBJECT = {
    "cat": "cat",
    "cat2": "cat",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
}

OBJECT_PROMPTS = [
    'a {} in the jungle',
    'a {} in the snow',
    'a {} on the beach',
    'a {} on a cobblestone street',
    'a {} on top of pink fabric',
    'a {} on top of a wooden floor',
    'a {} with a city in the background',
    'a {} with a mountain in the background',
    'a {} with a blue house in the background',
    'a {} on top of a purple rug in a forest',
    'a {} with a wheat field in the background',
    'a {} with a tree and autumn leaves in the background',
    'a {} with the Eiffel Tower in the background',
    'a {} floating on top of water',
    'a {} floating in an ocean of milk',
    'a {} on top of green grass with sunflowers around it',
    'a {} on top of a mirror',
    'a {} on top of the sidewalk in a crowded street',
    'a {} on top of a dirt road',
    'a {} on top of a white rug',
    'a red {}',
    'a purple {}',
    'a shiny {}',
    'a wet {}',
    'a cube shaped {}'
]

LIVE_OBJECT_PROMPTS = [
    'a {} in the jungle',
    'a {} in the snow',
    'a {} on the beach',
    'a {} on a cobblestone street',
    'a {} on top of pink fabric',
    'a {} on top of a wooden floor',
    'a {} with a city in the background',
    'a {} with a mountain in the background',
    'a {} with a blue house in the background',
    'a {} on top of a purple rug in a forest',
    'a {} wearing a red hat',
    'a {} wearing a santa hat',
    'a {} wearing a rainbow scarf',
    'a {} wearing a black top hat and a monocle',
    'a {} in a chef outfit',
    'a {} in a firefighter outfit',
    'a {} in a police outfit',
    'a {} wearing pink glasses',
    'a {} wearing a yellow shirt',
    'a {} in a purple wizard outfit',
    'a red {}',
    'a purple {}',
    'a shiny {}',
    'a wet {}',
    'a cube shaped {}'
]

KOSMOSG_OBJECT_PROMPTS = [
    '{} in the jungle',
    '{} in the snow',
    '{} on the beach',
    '{} on a cobblestone street',
    '{} on top of pink fabric',
    '{} on top of a wooden floor',
    '{} with a city in the background',
    '{} with a mountain in the background',
    '{} with a blue house in the background',
    '{} on top of a purple rug in a forest',
    '{} with a wheat field in the background',
    '{} with a tree and autumn leaves in the background',
    '{} with the Eiffel Tower in the background',
    '{} floating on top of water',
    '{} floating in an ocean of milk',
    '{} on top of green grass with sunflowers around it',
    '{} on top of a mirror',
    '{} on top of the sidewalk in a crowded street',
    '{} on top of a dirt road',
    '{} on top of a white rug',
    '{}, red',
    '{}, purple',
    '{}, shiny',
    '{}, wet',
    '{}, cube shaped'
]

KOSMOSG_LIVE_OBJECT_PROMPTS = [
    '{} in the jungle',
    '{} in the snow',
    '{} on the beach',
    '{} on a cobblestone street',
    '{} on top of pink fabric',
    '{} on top of a wooden floor',
    '{} with a city in the background',
    '{} with a mountain in the background',
    '{} with a blue house in the background',
    '{} on top of a purple rug in a forest',
    '{} wearing a red hat',
    '{} wearing a santa hat',
    '{} wearing a rainbow scarf',
    '{} wearing a black top hat and a monocle',
    '{} in a chef outfit',
    '{} in a firefighter outfit',
    '{} in a police outfit',
    '{} wearing pink glasses',
    '{} wearing a yellow shirt',
    '{} in a purple wizard outfit',
    '{}, red',
    '{}, purple',
    '{}, shiny',
    '{}, wet',
    '{}, cube shaped'
]


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
            # A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.0),
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
        item = self.get_sample(idx)
        return item
        # while(True):
        #     try:
        #         idx = np.random.randint(0, len(self.data)-1)
        #         item = self.get_sample(idx)
        #         return item
        #     except:
        #         idx = np.random.randint(0, len(self.data)-1)
                
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
    
    def draw_bboxes_all(self, tensor, boxes, color=(1.0, 0.0, 0.0), thickness=8):
        """
        在PyTorch图像张量上绘制边界框边框，针对[H, W, C]的张量布局。

        参数:
        - tensor: 图像张量，形状为(H, W, C)，H为高度，W为宽度，C为通道数。
        - boxes: 边界框列表，每个边界框的格式为[x_min, y_min, x_max, y_max]，坐标为归一化值。
        - color: 边框的颜色，默认为白色。
        - thickness: 边框的厚度。
        """
        H, W, C = tensor.shape
        
        x_min, y_min, x_max, y_max = boxes
        x_min, y_min, x_max, y_max = int(x_min * W), int(y_min * H), int(x_max * W), int(y_max * H)

        # 绘制顶部和底部
        tensor[y_min-int(thickness/2):y_min+int(thickness/2), x_min:x_max, :] = torch.tensor(color)
        tensor[y_max-int(thickness/2):y_max+ int(thickness/2), x_min:x_max, :] = torch.tensor(color)

        # 绘制左侧和右侧
        tensor[y_min:y_max, x_min-int(thickness/2):x_min+int(thickness/2), :] = torch.tensor(color)
        tensor[y_min:y_max, x_max-int(thickness/2):x_max+ int(thickness/2), :] = torch.tensor(color)

        return tensor
            
    def process_pairs_customized(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        assert mask_score(ref_mask) > 0.90
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.01, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑

        y1,y2,x1,x2 = ref_box_yyxx
        
        ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
        ref_image = pad_to_square(masked_ref_image, pad_value = 0)
        jpg = (cv2.resize(ref_image.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)/ 127.5 - 1.0).astype(np.float32) # [-1,1]之间
        
        ###################### 整图的bbox
        # ref_image = pad_to_square(ref_image, pad_value = 0)
        ref_mask_3_squared = pad_to_square(ref_mask_3, pad_value = 0)
        ref_mask_3_squared = cv2.resize(ref_mask_3_squared, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref_mask_squared =  ref_mask_3_squared[:,:,0]
        y1_new, y2_new, x1_new, x2_new = get_bbox_from_mask(ref_mask_squared)
        
        # layout = np.zeros((512,512,3), dtype=np.float32)  # 这里改颜色
        # layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 1.0, 1.0]
        layout = np.full((512, 512, 3), (150/255, 150/255, 150/255), dtype=np.float32)
        layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 0.0, 0.0] # 红色
        # layout = pad_to_square(layout, pad_value = 0, random = False)
        layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

        bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
        boxes = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 

        # layout_multi = torch.zeros(512, 512, 3)
        fill_value = torch.tensor([150/255, 150/255, 150/255], dtype=torch.float32)
        layout_multi = torch.ones((512, 512, 3), dtype=torch.float32) * fill_value

        layout_multi = self.draw_bboxes_all(layout_multi, bbox).cpu().numpy()  #只有一个box就不循环了

        ############################
        
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]  #(70,216,3) 只有该物体，将其抠出来. masked_ref_image1.png
        ref_mask = ref_mask[y1:y2,x1:x2]

        
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 0, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) #masked_ref_image3.png

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
        ref_mask = ref_mask_3[:,:,0]

        masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 
        masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        
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
        collage = np.zeros(cropped_target_image.shape, dtype=np.uint8)  #改成ones？
        # collage = cropped_target_image.copy()   #这里collage是否改为不要背景？ 

        collage[y1:y2,x1:x2,:] = ref_image_collage
        
        collage_mask = cropped_target_image.copy() * 0.0   #这里翻一下, mask掉背景? 应该不用改
        collage_mask[y1:y2,x1:x2,:] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]
        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32) #可以直接当做layout
        collage_mask[collage_mask == 2] = -1
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        # masked_ref_image_aug = masked_ref_image_aug  / 127.5 -1.0
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0         # [-1,1]之间
        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
        

        item = dict(
                ref=masked_ref_image_aug.copy(),  # masked_ref_image_aug.copy()（224,224,3）
                jpg=jpg.copy(),  # (512,512,3)
                layout = layout.copy(),
                boxes = boxes.copy(),
                layout_all = layout_multi.copy()
                ) 
        return item

class DreamBoothDataset(BaseDataset_t2i):
    def __init__(self, fg_dir, bg_dir, caption_dir):
        self.bg_dir = bg_dir
        bg_data = os.listdir(self.bg_dir)
        self.bg_data = [i for i in bg_data] #[i for i in bg_data if 'mask' in i]
        self.image_dir = fg_dir
        self.data  = os.listdir(self.image_dir)
        self.size = (512,512)
        self.clip_size = (224,224)
        self.caption_dir = caption_dir
        '''
         Dynamic:
            0: Static View, High Quality
            1: Multi-view, Low Quality
            2: Multi-view, High Quality
        '''
        self.dynamic = 1 
        self.max_boxes = 10

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])
        self.random_drop_embedding = 'none'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx = np.random.randint(0, len(self.data)-1)
        item = self.get_sample(idx)
        return item

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
        image = cv2.imread( mask_path) #, cv2.IMREAD_UNCHANGED
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask
    
    def get_sample(self, idx):
        
        dir_name = self.data[idx]
        dir_path = os.path.join(self.image_dir, dir_name)
        images = os.listdir(dir_path)
        image_name = [i for i in images if '.png' or 'jpg' in i][4] # or 'jpg'
        image_path = os.path.join(dir_path, image_name)

        image = cv2.imread( image_path, cv2.IMREAD_UNCHANGED)
        mask_path = image_path.replace('.jpg','_mask.png')
        mask = self.get_alpha_mask(mask_path)
        # mask = (image[:,:,-1] > 128).astype(np.uint8)
        # image = image[:,:,:-1]

        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        ref_image = image 
        ref_mask = mask
        
        
        # 用一样的图做bg就行
        tar_image = ref_image
        tar_mask = ref_mask

        item_with_collage = self.process_pairs_customized(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        #除了第一个其他都是0？

        array = torch.zeros(self.max_boxes) #np.ones(self.max_boxes,dtype=np.int32) # 从zeros改成ones，不mask了
        array[0] = 1
        item_with_collage['masks'] = array #.reshape(30,1)
        
        # import pdb ; pdb.set_trace()
        folder_name = image_path.split('/')[-2]
        object_name = folder_name
        # 或者这块用dict做匹配也行
        
        prompt = []
        if object_name in OBJECT:
            object_class = OBJECT[object_name]
            for i in OBJECT_PROMPTS:
                prompt.append(i.format(object_class))
            # prompt = OBJECT_PROMPTS[idx % 25]  #这块有问题，需要是一个才行
    
        else:
            object_class = LIVE_OBJECT[object_name]
            for i in LIVE_OBJECT_PROMPTS:
                prompt.append(i.format(object_class))
            # prompt = LIVE_OBJECT_PROMPTS[idx % 25]
            
       
        class_key = class2id[folder_name]

        item_with_collage['positive'] = class_key
        
       

        prompt_list = []
        prompt_list.append(class_key)
        prompt_list.extend([''] * (self.max_boxes - 1))
        item_with_collage['positive_all'] = ','.join(prompt_list)
        
        item_with_collage['anno_id'] = image_path  # target image path
        item_with_collage['txt'] = prompt

       
        image_crop = self.preprocess(item_with_collage['ref']).float()
        item_with_collage['ref_processed'] = image_crop
        # print(item_with_collage['ref'].dtype())
        item_with_collage['ref'] = item_with_collage['ref'].transpose(2,0,1).astype(np.float32)

        # add random drop

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(item_with_collage['masks'], self.random_drop_embedding)
        else:
            image_masks = item_with_collage['masks']
            text_masks = item_with_collage['masks']

        item_with_collage["text_masks"] = text_masks  # item_with_collage['txt']
        item_with_collage["image_masks"] = image_masks #item_with_collage['txt']
        
        item_with_collage["caption"] = item_with_collage['txt'] 
        item_with_collage['box_ref'] = item_with_collage['boxes'][0:1]  #.unsqueeze(1)
        item_with_collage["image"] = item_with_collage["jpg"].transpose(2,0,1) #.transpose(1,2,0)
        item_with_collage["object_name"] = object_name
        return item_with_collage
