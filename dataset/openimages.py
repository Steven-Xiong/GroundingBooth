
from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd
#### This is borrow from ELITE
from .data_utils1 import * 
from tqdm import tqdm
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()

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
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=90, p=0.8),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        # import pdb; pdb.set_trace()
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
        # item = self.get_sample(idx)
        # return item
        # try:
        #     item = self.get_sample(idx)
        #     return item
        # except:
        #     idx = np.random.randint(0, len(self.data)-1)
            
        while(True):
            try:
                idx = np.random.randint(0, len(self.data)-1) #这一句不要就可以做测试了
                item = self.get_sample(idx)
                return item
            except:
                # idx = np.random.randint(0, len(self.data)-1)
                return None
                
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
    
    def process_boxes(self,ref_image, anno_list, max_ratio = 0.8):
        boxes = []
        layout_list = []
        layout_multi = torch.zeros(512, 512, 3)
        # import pdb; pdb.set_trace()
        for anno in anno_list:
            # anno = anno_list[idx]
            ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
            tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
            # assert mask_score(ref_mask) > 0.90   # 0.90 改低一点，增加训练数量
            # assert self.check_mask_area(ref_mask) == True
            # assert self.check_mask_area(tar_mask)  == True
            ref_box_yyxx = get_bbox_from_mask(ref_mask)
            # print('ref_box_yyxx',ref_box_yyxx)
            # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.05, mode = 'min') == True
            # Filtering background for the reference image
            ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
            masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑
            y1,y2,x1,x2 = ref_box_yyxx

            ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
            # tar_box_yyxx_crop = box2squre(tar_image, ref_box_yyxx)
            # import pdb; pdb.set_trace()
            ref_image_squared = pad_to_square(ref_image, pad_value = 0)  #7.25 修改
            ref_mask_3_squared = pad_to_square(ref_mask_3, pad_value = 0)
            ref_mask_3_squared = cv2.resize(ref_mask_3_squared, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
            ref_mask_squared =  ref_mask_3_squared[:,:,0]
            y1_new, y2_new, x1_new, x2_new = get_bbox_from_mask(ref_mask_squared)
            bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
            boxes.append(bbox)

            '''
            tar_box_yyxx_squared = box2squre(ref_image, ref_box_yyxx) # 保证box不超出编辑
            ratio = 512 / ref_image.shape[0]
            y1_new,y2_new,x1_new,x2_new = tar_box_yyxx_squared 
            y1_new,y2_new,x1_new,x2_new = int(y1_new * ratio), int(y2_new * ratio),int(x1_new*ratio),int(x2_new * ratio)
            bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
            boxes.append(bbox)
            '''
            # generate the whole layout
            # layout = np.zeros((image_tensor.shape[1], image_tensor.shape[2], 3), dtype=np.float32)
            # layout[int(box[1]*self.image_size):int(box[3]*self.image_size), int(box[0]*self.image_size):int(box[2]*self.image_size)] = [1.0, 1.0, 1.0]
            layout = np.zeros((512,512,3), dtype=np.float32)
            layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 1.0, 1.0]
            layout_list.append(layout)
            layout_multi = self.draw_bboxes_all(layout_multi, bbox)

        # import pdb; pdb.set_trace()
        if len(boxes) < self.max_boxes:
            boxes.extend([np.array([0, 0, 0, 0])] * (self.max_boxes - len(boxes)))
        elif len(boxes) >= self.max_boxes:
            boxes = boxes[:self.max_boxes]
        boxes = np.stack(boxes)  
        
        if len(layout_list) != 0:
            # out['layout'] = torch.from_numpy(layout_all[0])
            layout_all = layout_multi.cpu().numpy()
            layout = layout_list[0]
        else:
            # out['layout'] = torch.zeros([512,512,3])
            layout_all = np.zeros((512, 512, 3), dtype=np.float32) #torch.zeros([512,512,3]) #
            layout = np.zeros((512, 512, 3), dtype=np.float32)
        return boxes, layout_all, layout


    def process_pairs_customized(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        # import pdb; pdb.set_trace()
        # assert mask_score(ref_mask) > 0.50   # 0.90 改低一点，增加训练数量
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        # import pdb; pdb.set_trace()
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.01, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑

        y1,y2,x1,x2 = ref_box_yyxx

        ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
        # import pdb; pdb.set_trace()
        # tar_box_yyxx_crop = box2squre(tar_image, ref_box_yyxx)
        ref_image = pad_to_square(ref_image, pad_value = 0)
        # tar_box_yyxx_squared = box2squre(ref_image, ref_box_yyxx) # square之后的bbox坐标
        
        # ratio = 512 / ref_image.shape[0]
        # y1_new,y2_new,x1_new,x2_new = tar_box_yyxx_squared 
        # y1_new,y2_new,x1_new,x2_new = int(y1_new * ratio), int(y2_new * ratio),int(x1_new*ratio),int(x2_new * ratio)

        # layout = np.zeros((512,512,3), dtype=np.float32)
        # layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 1.0, 1.0]
        # # layout = pad_to_square(layout, pad_value = 0, random = False)
        # layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        jpg = (cv2.resize(ref_image.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)/ 127.5 - 1.0).astype(np.float32) # [-1,1]之间
        # [-1,1]之间
        
        ######################
        
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]  #(70,216,3) 只有该物体，将其抠出来. masked_ref_image1.png
        ref_mask = ref_mask[y1:y2,x1:x2]
        # 5.30 这里不要做expand了  7.11 重新做？
        # if self.mode == 'train':
        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        # else:
        #     pass
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
        # masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        # masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 
        
        masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        # if self.mode == 'train':
        #     masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        # else:
        #     masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 

        masked_ref_image_aug = masked_ref_image_compose.copy()
        '''
        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        # ========= Training Target ===========
        # import pdb; pdb.set_trace()
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        # 6.6 不做expand
        # tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        # assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        
        # Cropping around the target object 
        # 6.6 不做expand
        # tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])  
        tar_box_yyxx_crop = tar_box_yyxx 
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx

        # import pdb; pdb.set_trace()
        layout = np.zeros((tar_box_yyxx_crop[1]-tar_box_yyxx_crop[0], tar_box_yyxx_crop[3]-tar_box_yyxx_crop[2], 3), dtype=np.float32)
        layout[y1:y2,x1:x2,:] = [1.0, 1.0, 1.0]
        layout = pad_to_square(layout, pad_value = 0, random = False)
        layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

        white_pixels = np.where(layout == 1.0)
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

        bbox = np.array([xmin, ymin, xmax, ymax])
        boxes = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 

        #/255 #-1.0
        
        # Prepairing collage image 保证这是square就行
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        import pdb; pdb.set_trace()
        collage = np.zeros(cropped_target_image.shape, dtype=np.uint8)  #改成ones？
        # collage = cropped_target_image.copy()   #这里collage是否改为不要背景？ 
        
        # if x2-x1 == ref_image_collage.shape[1] or y2-y1 == ref_image_collage.shape[0]:
        try:
            collage[y1:y2,x1:x2,:] = ref_image_collage
            collage_mask = cropped_target_image.copy() * 0.0   #这里翻一下, mask掉背景? 应该不用改
            collage_mask[y1:y2,x1:x2,:] = 1.0
        except:
            collage[y1:y1+ref_image_collage.shape[0],x1:x1+ref_image_collage.shape[1],:] = ref_image_collage
            collage_mask = cropped_target_image.copy() * 0.0   #这里翻一下, mask掉背景? 应该不用改
            collage_mask[y1:y1+ref_image_collage.shape[0],x1:x1+ref_image_collage.shape[1],:] = 1.0
        

        # if np.random.uniform(0, 1) < 0.7: 
        #     cropped_tar_mask = perturb_mask(cropped_tar_mask)
        #     collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

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
        
        # masked_ref_image_aug = masked_ref_image_aug  / 127.5 -1.0
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0         # [-1,1]之间
        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)   

        '''
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        ref=masked_ref_image_aug.copy(),  # masked_ref_image_aug.copy()（224,224,3）
        jpg=jpg.copy(),  # (512,512,3)
        # layout = layout.copy(),
        # tar_box_yyxx_crop=np.array(tar_box_yyxx_crop), 
        # boxes = boxes.copy()
        
        item = dict(
                ref = ref[0],  # (512,512,3)
                jpg = jpg[0]
                ) 
        return item

    def process_boxes_openimage(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        # assert mask_score(ref_mask) > 0.90
        # import pdb; pdb.set_trace()
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        # import pdb; pdb.set_trace()
        boxes = []
        layout_list = []
        layout_multi = torch.zeros(512, 512, 3)
        
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.01, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑
        # 这里控制，如果ref object在裁剪区域之内就用crop，如果在之外就pad
        
        y1,y2,x1,x2 = ref_box_yyxx
        # print('ref_box_yyxx_before:',ref_box_yyxx)
        
        
        ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
        # tar_box_yyxx_crop = box2squre(tar_image, ref_box_yyxx)
        # import pdb; pdb.set_trace()
        ref_image_squared = pad_to_square(ref_image, pad_value = 0)  #7.25 修改
        ref_mask_3_squared = pad_to_square(ref_mask_3, pad_value = 0)
        ref_mask_3_squared = cv2.resize(ref_mask_3_squared, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref_mask_squared =  ref_mask_3_squared[:,:,0]
        y1_new, y2_new, x1_new, x2_new = get_bbox_from_mask(ref_mask_squared)
        bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
        boxes.append(bbox)

        '''
        tar_box_yyxx_squared = box2squre(ref_image, ref_box_yyxx) # 保证box不超出编辑
        ratio = 512 / ref_image.shape[0]
        y1_new,y2_new,x1_new,x2_new = tar_box_yyxx_squared 
        y1_new,y2_new,x1_new,x2_new = int(y1_new * ratio), int(y2_new * ratio),int(x1_new*ratio),int(x2_new * ratio)
        bbox = np.array([x1_new/512, y1_new/512, x2_new/512, y2_new/512])
        boxes.append(bbox)
        '''
        # generate the whole layout
        # layout = np.zeros((image_tensor.shape[1], image_tensor.shape[2], 3), dtype=np.float32)
        # layout[int(box[1]*self.image_size):int(box[3]*self.image_size), int(box[0]*self.image_size):int(box[2]*self.image_size)] = [1.0, 1.0, 1.0]
        layout = np.zeros((512,512,3), dtype=np.float32)
        layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 1.0, 1.0]
        layout_list.append(layout)
        layout_multi = self.draw_bboxes_all(layout_multi, bbox)
        
        if len(boxes) < self.max_boxes:
            boxes.extend([np.array([0, 0, 0, 0])] * (self.max_boxes - len(boxes)))
        elif len(boxes) >= self.max_boxes:
            boxes = boxes[:self.max_boxes]
        boxes = np.stack(boxes)  
        
        if len(layout_list) != 0:
            # out['layout'] = torch.from_numpy(layout_all[0])
            layout_all = layout_multi.cpu().numpy()
            layout = layout_list[0]
        else:
            # out['layout'] = torch.zeros([512,512,3])
            layout_all = np.zeros((512, 512, 3), dtype=np.float32) #torch.zeros([512,512,3]) #
            layout = np.zeros((512, 512, 3), dtype=np.float32)

        # item = dict(
        #         layout_all=layout_all.copy(),  # (512,512,3)
        #         layout = layout.copy(),
        #         boxes = boxes.copy()
        #         ) 
        return layout_all, layout, boxes
    
        # return boxes, layout_all, layout
    
    def process_pairs_customized_mvimagenet(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        # assert mask_score(ref_mask) > 0.90
        # import pdb; pdb.set_trace()
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        # import pdb; pdb.set_trace()
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.01, mode = 'min') == True
        
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
        # if self.mode == 'train':
        #     masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        # else:
        #     masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 
            
        masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        
        masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        # ========= Training Target ===========
        # import pdb; pdb.set_trace()
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        
        
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

        # import pdb; pdb.set_trace()
        layout = np.zeros((tar_box_yyxx_crop[1]-tar_box_yyxx_crop[0], tar_box_yyxx_crop[3]-tar_box_yyxx_crop[2], 3), dtype=np.float32)
        layout[y1:y2,x1:x2,:] = [1.0, 1.0, 1.0]
        layout = pad_to_square(layout, pad_value = 0, random = False)
        layout = cv2.resize(layout.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

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

         #/255 #-1.0
        
        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        # import pdb; pdb.set_trace()
        collage = np.zeros(cropped_target_image.shape, dtype=np.uint8)  #改成ones？
        collage[y1:y2,x1:x2,:] = ref_image_collage   #这里collage是否改为不要背景？ 
        # try:
        #     collage[y1:y2,x1:x2,:] = ref_image_collage
        # except:
        #     collage[y1:y2,x1:x2+1,:] = ref_image_collage
        #     print('collage[y1:y2,x1:x2,:]', collage[y1:y2,x1:x2,:].shape)
        #     print('ref_image_collage', ref_image_collage.shape)
        
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
        # import pdb; pdb.set_trace()
        

        item = dict(
                ref=masked_ref_image_aug.copy(),  # masked_ref_image_aug.copy()（224,224,3）
                jpg=cropped_target_image.copy(),  # (512,512,3)
                # tar_box_yyxx_crop=np.array(tar_box_yyxx_crop), 
                layout = layout.copy(),
                boxes = boxes.copy()
                ) 
        return item
        
class OpenImagesDataset(BaseDataset_t2i):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        set="train",
        placeholder_token="*",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.set_type = set

        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

        self.bbox_path_list = []
        #### data root: 'data/Open_Imagesv7'
        
        if set == "train":
            bboxs_path = os.path.join(data_root, 'annotations', f'oidv6-train-annotations-bbox.csv')
        elif set == "validation":
            bboxs_path = os.path.join(data_root, 'annotations', f'validation-annotations-bbox.csv')
        else:
            bboxs_path = os.path.join(data_root, 'annotations', f'test-annotations-bbox.csv')

        df_val_bbox = pd.read_csv(bboxs_path)
        bbox_groups = df_val_bbox.groupby(df_val_bbox.LabelName)

        bbox_full = []
        for label_name in df_val_bbox['LabelName'].unique():
            bboxs = bbox_groups.get_group(label_name)[
                ['XMin', 'XMax', 'YMin', 'YMax', 'LabelName', 'ImageID',
                 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsInside']].values.tolist()
            bboxs_new = []
            for bbox in bboxs:
                if not ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) > 0.8 or (bbox[1] - bbox[0]) * (
                        bbox[3] - bbox[2]) < 0.02):
                    bboxs_new.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
            bbox_full.extend(bboxs_new)

        self.bboxs_full = bbox_full

        self.num_images = len(bbox_full)

        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small


    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = np.array(image)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def obtain_text(self, add_caption, object_category=None):

        if object_category is None:
            placeholder_string = self.placeholder_token
        else:
            placeholder_string = object_category

        text = random.choice(self.templates).format(placeholder_string)
        text = add_caption + text[1:]

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        index = torch.tensor(placeholder_index)

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return input_ids, index, text

    def __getitem__(self, i):
        example = {}
        # import pdb; pdb.set_trace()
        input_ids, index, text = self.obtain_text('a')
        example["input_ids"] = input_ids
        example["index"] = index
        example["text"] = text

        bbox_sample = self.bboxs_full[i % self.num_images]
        bbox_sample = copy.copy(bbox_sample)

        file_name = bbox_sample[-1] + '.jpg'
        img_path = os.path.join(self.data_root, 'images', self.set_type, file_name)

        try:
            img_p = Image.open(img_path).convert("RGB")
            img_p_np = np.array(img_p)
            bbox_sample[0] *= int(img_p_np.shape[1])
            bbox_sample[1] *= int(img_p_np.shape[1])
            bbox_sample[2] *= int(img_p_np.shape[0])
            bbox_sample[3] *= int(img_p_np.shape[0])

            bbox_pad = copy.copy(bbox_sample)
            bbox_pad[0] = int(bbox_sample[0] - min(10, bbox_sample[0] - 0))
            bbox_pad[1] = int(bbox_sample[1] + min(10, img_p.size[0] - bbox_sample[1]))
            bbox_pad[2] = int(bbox_sample[2] - min(10, bbox_sample[2] - 0))
            bbox_pad[3] = int(bbox_sample[3] + min(10, img_p.size[1] - bbox_sample[3]))

            image_tensor = img_p_np[bbox_pad[2]:bbox_pad[3], bbox_pad[0]:bbox_pad[1], :]
            example["pixel_values"] = self.process(image_tensor)

            ref_image_tensor = self.random_trans(image=image_tensor)
            ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)

        except Exception as e:
            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            with open('error.txt', 'a+') as f:
                f.write(str(e) + '\n')

        return example




# 用这个就行了
class OpenImagesDatasetWithMask(OpenImagesDataset):
    def __init__(self,
             data_root,
             tokenizer,
             size=512,
             interpolation="bicubic",
             set="train",
             placeholder_token="*"):

        # super().__init__(data_root, tokenizer, size, interpolation, set, placeholder_token)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.set = set
        # import pdb; pdb.set_trace()
        class_anno_path = os.path.join(data_root, 'annotations', f'oidv7-class-descriptions.csv')
        anno_files = pd.read_csv(class_anno_path)
        class_groups = anno_files.groupby(anno_files.LabelName)

        if set == "train":
            bboxs_path = os.path.join(data_root, 'annotations', f'train-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'train_bbox_dict.npy')
        elif set == "validation":
            bboxs_path = os.path.join(data_root, 'annotations', f'validation-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'validation_bbox_dict.npy')
        else:
            bboxs_path = os.path.join(data_root, 'annotations', f'test-annotations-object-segmentation.csv')
            dict_path = os.path.join(data_root, 'segs', f'test_bbox_dict.npy')

        bbox_dict = np.load(dict_path, allow_pickle=True).item()

        df_val_bbox = pd.read_csv(bboxs_path)
        bbox_groups = df_val_bbox.groupby(df_val_bbox.LabelName)
        bboxes_full = []
        for label_name in df_val_bbox['LabelName'].unique():
            bboxs = bbox_groups.get_group(label_name)[
                ['BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax', 'LabelName', 'MaskPath']].values.tolist()
            bboxes_new = []
            for box in bboxs:
                if not box[-1] in bbox_dict:
                    continue
                bbox_data = bbox_dict[box[-1]]

                if (bbox_data[2] - bbox_data[1]) < 100 or (bbox_data[4] - bbox_data[3]) < 100:
                    continue
                if not ((bbox_data[2] - bbox_data[1]) / (bbox_data[4] - bbox_data[3]) < 0.5 or (
                        bbox_data[4] - bbox_data[3]) / ( bbox_data[2] - bbox_data[1]) < 0.5):
                    class_name = class_groups.get_group(box[4])[['DisplayName']].values.tolist()[0][0]
                    bboxes_new.append([box[-1], bbox_data[1], bbox_data[2], bbox_data[3], bbox_data[4], class_name])

            bboxes_full.extend(bboxes_new)

        self.bboxes_full = bboxes_full
        self.num_images = len(bboxes_full)

        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small
        
        self.random_drop_embedding = 'both'
        self.prob_use_caption = 0.9
        self.prob_use_ref = 0.9
        # dropout
        # if self.set == 'train':
        #     self.random_drop_embedding = 'both'
        #     self.prob_use_caption = 0.9
        #     self.prob_use_ref = 0.9
        # else:
        # self.random_drop_embedding = 'none'
        # self.prob_use_caption = 1.0
        # self.prob_use_ref = 1.0
        
        # import pdb; pdb.set_trace()
        self.max_boxes = 10
        self.mode = self.set

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])
        
        self.dynamic = 2

    def __len__(self):
        return self._length

    ## borrowed from custom diffusion
    def custom_aug(self, instance_image):
        instance_image = Image.fromarray(instance_image)
        #### apply augmentation and create a valid image regions mask ####
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(["a far away", "very small"])
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2,
            cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
            (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.

        elif random_scale > self.size:
            add_to_caption = np.random.choice(["zoomed in", "close up"])
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                             cy - self.size // 2: cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            add_to_caption = "a"
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))

        return torch.from_numpy(instance_image).permute(2, 0, 1), torch.from_numpy(mask[:, :, None]).permute(2, 0, 1), add_to_caption

    def aug_cv2(self, img, seg):

        img_auged = np.array(img).copy()
        seg_auged = np.array(seg).copy()
        # resize and crop
        if random.choice([0, 1]) == 0:
            new_size = random.randint(224, 256)
            img_auged = cv2.resize(img_auged, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            seg_auged = cv2.resize(seg_auged, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

            start_x, start_y = random.randint(0, new_size - 224), random.randint(0, new_size - 224)
            img_auged = img_auged[start_x:start_x + 224, start_y:start_y + 224, :]
            seg_auged = seg_auged[start_x:start_x + 224, start_y:start_y + 224, :]

        h, w = img_auged.shape[:2]
        # rotate
        if random.choice([0, 1]) == 0:
            # print('rotate')
            angle = random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1)
            img_auged = cv2.warpAffine(img_auged, M, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, M, (w, h), flags=cv2.INTER_NEAREST)

        # translation
        if random.choice([0, 1]) == 0:
            trans_x = random.randint(-60, 60)
            trans_y = random.randint(-60, 60)
            H = np.float32([[1, 0, trans_x],
                            [0, 1, trans_y]])
            img_auged = cv2.warpAffine(img_auged, H, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, H, (w, h), flags=cv2.INTER_NEAREST)

        img_auged = Image.fromarray(img_auged)
        seg_auged = Image.fromarray(seg_auged)

        return img_auged, seg_auged


    def __getitem__(self, i):
        example = {}
        # import pdb; pdb.set_trace()
        seg_name = self.bboxes_full[i % self.num_images][0]
        file_name = seg_name.split('_')[0] + '.jpg'
        img_path = os.path.join(self.data_root, 'images', self.set, file_name)
        seg_path = os.path.join(self.data_root, 'segs', self.set, seg_name)

        # try:
        # crop image and mask
        bbox_sample = self.bboxes_full[i % self.num_images][1:]
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        seg_p_np = cv2.imread(seg_path).astype('float')
        seg_p_np = cv2.resize(seg_p_np, img_p_np.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        bbox_pad = copy.copy(bbox_sample)
        pad_size = random.choice(list(range(10, 20)))
        bbox_pad[0] = int(bbox_pad[0] - min(pad_size, bbox_pad[0] - 0))
        bbox_pad[1] = int(bbox_pad[1] + pad_size)
        bbox_pad[2] = int(bbox_pad[2] - min(pad_size, bbox_pad[2] - 0))
        bbox_pad[3] = int(bbox_pad[3] + pad_size)

        image_tensor = img_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]
        seg_tensor = seg_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]

        # augmentation for input image
        augged_image, augged_mask, add_caption = self.custom_aug(image_tensor)
        input_ids, index, text = self.obtain_text(add_caption)
        # import pdb; pdb.set_trace()

        
        # subject class
        class_key = bbox_sample[-1]  #'suit'
        example['positive'] = class_key
        prompt_list = []
        prompt_list.append(class_key)
        prompt_list.extend([''] * (self.max_boxes - 1))
        example['positive_all'] = ','.join(prompt_list)  # mvimgnet只用一个object

        # example["image"] = augged_image    # tensor [3,512,512]
        # example["mask_values"] = augged_mask
        # example["input_ids"] = input_ids
        # example["index"] = index
        example["txt"] = text.replace('*', class_key)

        object_tensor = image_tensor * (seg_tensor / 255)
        ref_object_tensor = cv2.resize(object_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        ref_image_tenser = cv2.resize(image_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        ref_seg_tensor = cv2.resize(seg_tensor, (224, 224), interpolation=cv2.INTER_NEAREST)
        # import pdb; pdb.set_trace()
        seg_mask = (seg_tensor[..., 0] // 255).astype(np.uint8)
        # items = self.process_pairs_customized_mvimagenet(image_tensor, seg_mask, image_tensor, seg_mask)  # add layout    #(500,333,3)
        
        
        items = self.process_pairs_customized(image_tensor, seg_mask, image_tensor, seg_mask)  # add layout    #(500,333,3)
        items['layout_all'], items['layout'],items['boxes'] = self.process_boxes_openimage(image_tensor, seg_mask, image_tensor, seg_mask)  
        
        sampled_time_steps = self.sample_timestep()
        example['time_steps'] = sampled_time_steps
        
        # import pdb; pdb.set_trace()
        example['ref'] = items['ref']
        example['jpg'] = items['jpg']
        example['layout'] = items['layout']
        example['layout_all'] = items['layout']
        example['boxes'] = items['boxes']
        example['ref_processed'] = self.preprocess(example['ref']).float()
        example['ref'] = example['ref'].transpose(2,0,1).astype(np.float32)

        ref_object_tensor, ref_seg_tensor = self.aug_cv2(ref_object_tensor.astype('uint8'), ref_seg_tensor.astype('uint8'))
        # example["pixel_values_clip"] = self.get_tensor_clip()(Image.fromarray(ref_image_tenser))
        # example["pixel_values_ref"] = self.get_tensor_clip()(ref_object_tensor)
        # example["pixel_values_seg"] = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
        

        #除了第一个其他都是0？
        array = torch.zeros(self.max_boxes) #np.zeros(self.max_boxes,dtype=np.int32)
        array[0] = 1
        example['masks'] = array #.reshape(30,1)


        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(example['masks'], self.random_drop_embedding)
        else:
            image_masks = example['masks']
            text_masks = example['masks']
        

        example["text_masks"] = text_masks  # item_with_collage['txt']
        example["image_masks"] = image_masks #item_with_collage['txt']

        if random.uniform(0, 1) < self.prob_use_caption:
            example["caption"] = example["txt"]   # 注意：这里是带placeholder的
        else:
            example["caption"] = ""
        if random.uniform(0, 1) < self.prob_use_ref:
            pass
        else:
            example["ref"] = np.zeros((3,224,224), dtype=np.float32)#.astype(np.float32)
        
        example['box_ref'] = example['boxes'][0:1]  #.unsqueeze(1)
        example["image"] = example["jpg"].transpose(2,0,1) #.transpose(1,2,0)
        
        example = convert_np_to_tensor(example)
        # except Exception as e:
        #     example["pixel_values"] = torch.zeros((3, 512, 512))
        #     example["pixel_values_obj"] = torch.zeros((3, 224, 224))
        #     example["pixel_values_clip"] = torch.zeros((3, 224, 224))
        #     example["pixel_values_seg"] = torch.zeros((3, 224, 224))

        #     input_ids, index, text = self.obtain_text("a")
        #     example["input_ids"] = input_ids
        #     example["index"] = index
        #     example["text"] = text

        #     with open('error.txt', 'a+') as f:
        #         f.write(str(e) + '\n')

        return example


class CustomDatasetWithBG(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token

        self.image_paths = []
        self.image_paths += [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if is_image(file_path) and not 'bg' in file_path]

        self.image_paths = sorted(self.image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = template

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):
        example = {}

        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image = Image.open(self.image_paths[i % self.num_images])

        mask_path = self.image_paths[i % self.num_images].replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
        mask = np.array(Image.open(mask_path))

        mask = np.where(mask > 0, 1, 0)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)
        object_tensor = image_np * mask
        example["pixel_values"] = self.process(image_np)


        ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
        ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
        example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
        example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
        example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

        return example


def compute_bbox(seg_path, image_path):
    """
    根据分割图和对应的图像计算 bbox 信息。
    返回：[cntr, hs, he, ws, we] 或 None（失败时）。
    """
    seg = cv2.imread(seg_path)
    image = cv2.imread(image_path)
    if seg is None or image is None:
        return None
    # 调整 seg 尺寸为 image 的尺寸
    seg = cv2.resize(seg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 只取第一通道
    seg = seg[:, :, 0]
    
    # 获得轮廓
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) == 0:
        return None
    cntr = np.vstack(contours) if len(contours) > 1 else contours[0]
    if len(cntr) < 2:
        return None

    hs, he = np.min(cntr[:, :, 1]), np.max(cntr[:, :, 1])
    ws, we = np.min(cntr[:, :, 0]), np.max(cntr[:, :, 0])
    h, w = seg.shape

    # 调整边界使宽高为偶数
    if (he - hs) % 2 == 1 and (he + 1) <= h:
        he += 1
    if (he - hs) % 2 == 1 and (hs - 1) >= 0:
        hs -= 1
    if (we - ws) % 2 == 1 and (we + 1) <= w:
        we += 1
    if (we - ws) % 2 == 1 and (ws - 1) >= 0:
        ws -= 1

    if he - hs < 2 or we - ws < 2:
        return None

    return [cntr, hs, he, ws, we]

class OpenImagesDatasetWithMask_train(OpenImagesDataset):
    def __init__(self,
                 data_root,
                 tokenizer,
                 size=512,
                 interpolation="bicubic",
                 set="train",
                 placeholder_token="*"):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.set = set

        # 读取类别描述文件，便于后续获取类别显示名称
        class_anno_path = os.path.join(data_root, 'annotations', 'oidv7-class-descriptions.csv')
        anno_files = pd.read_csv(class_anno_path)
        self.class_groups = anno_files.groupby('LabelName')

        # 根据 set 选择对应的 CSV 注释文件
        if set == "train":
            bboxs_path = os.path.join(data_root, 'annotations', 'train-annotations-object-segmentation.csv')
        elif set == "validation":
            bboxs_path = os.path.join(data_root, 'annotations', 'validation-annotations-object-segmentation.csv')
        else:
            bboxs_path = os.path.join(data_root, 'annotations', 'test-annotations-object-segmentation.csv')
        df_bbox = pd.read_csv(bboxs_path)
        
        # 保存 CSV 中需要的字段到列表，延后在 __getitem__ 中实时处理 bbox
        self.bbox_rows = df_bbox[['BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax', 'LabelName', 'MaskPath']].values.tolist()
        self.num_images = len(self.bbox_rows)
        print(f"{set}: total {self.num_images} images ...")

        self.image_dir = os.path.join(data_root, 'images', set)
        self.seg_dir = os.path.join(data_root, 'segs', set)

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small
        
        if self.set == 'train':
            self.random_drop_embedding = 'both'
            self.prob_use_caption = 0.9
            self.prob_use_ref = 0.9
        else:
            self.random_drop_embedding = 'none'
            self.prob_use_caption = 1.0
            self.prob_use_ref = 1.0
        
        self.max_boxes = 10
        self.mode = self.set

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.dynamic = 2

    def __len__(self):
        return self.num_images

    def custom_aug(self, instance_image):
        instance_image = Image.fromarray(instance_image)
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
        if random_scale % 2 == 1:
            random_scale += 1
        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(["a far away", "very small"])
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)
            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2,
                           cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1
            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
                 (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
        elif random_scale > self.size:
            add_to_caption = np.random.choice(["zoomed in", "close up"])
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                                            cy - self.size // 2: cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            add_to_caption = "a"
            instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))
        return torch.from_numpy(instance_image).permute(2, 0, 1), torch.from_numpy(mask[:, :, None]).permute(2, 0, 1), add_to_caption

    def aug_cv2(self, img, seg):
        img_auged = np.array(img).copy()
        seg_auged = np.array(seg).copy()
        if random.choice([0, 1]) == 0:
            new_size = random.randint(224, 256)
            img_auged = cv2.resize(img_auged, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            seg_auged = cv2.resize(seg_auged, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
            start_x, start_y = random.randint(0, new_size - 224), random.randint(0, new_size - 224)
            img_auged = img_auged[start_x:start_x + 224, start_y:start_y + 224, :]
            seg_auged = seg_auged[start_x:start_x + 224, start_y:start_y + 224, :]
        h, w = img_auged.shape[:2]
        if random.choice([0, 1]) == 0:
            angle = random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1)
            img_auged = cv2.warpAffine(img_auged, M, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, M, (w, h), flags=cv2.INTER_NEAREST)
        if random.choice([0, 1]) == 0:
            trans_x = random.randint(-60, 60)
            trans_y = random.randint(-60, 60)
            H = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            img_auged = cv2.warpAffine(img_auged, H, (w, h), flags=cv2.INTER_CUBIC)
            seg_auged = cv2.warpAffine(seg_auged, H, (w, h), flags=cv2.INTER_NEAREST)
        img_auged = Image.fromarray(img_auged)
        seg_auged = Image.fromarray(seg_auged)
        return img_auged, seg_auged

    def __getitem__(self, i):
        # 先取出当前样本在 CSV 中的信息
        row = self.bbox_rows[i % self.num_images]
        label_name = row[4]
        mask_name = row[5]
        file_name = mask_name.split('_')[0] + '.jpg'
        img_path = os.path.join(self.image_dir, file_name)
        seg_path = os.path.join(self.seg_dir, mask_name)

        # 实时计算 bbox（compute_bbox 返回 [cntr, hs, he, ws, we]）
        bbox = compute_bbox(seg_path, img_path)
        if bbox is None:
            # 如果计算失败，则尝试取下一个样本
            return self.__getitem__((i + 1) % self.num_images)
        bbox_sample = bbox[1:]  # [hs, he, ws, we]

        # 读取 image 与 segmentation，并统一调整大小
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        seg_p_np = cv2.imread(seg_path).astype('float')
        seg_p_np = cv2.resize(seg_p_np, img_p_np.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # 对 bbox 进行随机扩展
        bbox_pad = copy.copy(bbox_sample)
        pad_size = random.choice(list(range(10, 20)))
        bbox_pad[0] = int(bbox_pad[0] - min(pad_size, bbox_pad[0]))
        bbox_pad[1] = int(bbox_pad[1] + pad_size)
        bbox_pad[2] = int(bbox_pad[2] - min(pad_size, bbox_pad[2]))
        bbox_pad[3] = int(bbox_pad[3] + pad_size)

        # 裁剪出目标区域
        image_tensor = img_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]
        seg_tensor = seg_p_np[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]

        # 接下来调用自定义 augmentation 及文本处理（obtain_text、process_pairs_customized 等）
        augged_image, augged_mask, add_caption = self.custom_aug(image_tensor)
        input_ids, index, text = self.obtain_text(add_caption)

        # 使用类别信息，若有 display name 则取 display name
        if label_name in self.class_groups.groups:
            class_disp = self.class_groups.get_group(label_name)['DisplayName'].values[0]
        else:
            class_disp = label_name
        example = {}
        example['positive'] = class_disp
        prompt_list = [class_disp] + [''] * (self.max_boxes - 1)
        example['positive_all'] = ','.join(prompt_list)
        example["txt"] = text.replace('*', class_disp)

        object_tensor = image_tensor * (seg_tensor / 255)
        ref_object_tensor = cv2.resize(object_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        ref_image_tenser = cv2.resize(image_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        ref_seg_tensor = cv2.resize(seg_tensor, (224, 224), interpolation=cv2.INTER_NEAREST)
        seg_mask = (seg_tensor[..., 0] // 255).astype(np.uint8)
        
        items = self.process_pairs_customized(image_tensor, seg_mask, image_tensor, seg_mask)
        items['layout_all'], items['layout'], items['boxes'] = self.process_boxes_openimage(image_tensor, seg_mask, image_tensor, seg_mask)
        sampled_time_steps = self.sample_timestep()
        example['time_steps'] = sampled_time_steps

        example['ref'] = items['ref']
        example['jpg'] = items['jpg']
        example['layout'] = items['layout']
        example['layout_all'] = items['layout']
        example['boxes'] = items['boxes']
        example['ref_processed'] = self.preprocess(example['ref']).float()
        example['ref'] = example['ref'].transpose(2, 0, 1).astype(np.float32)

        ref_object_tensor, ref_seg_tensor = self.aug_cv2(ref_object_tensor.astype('uint8'), ref_seg_tensor.astype('uint8'))
        array = torch.zeros(self.max_boxes)
        array[0] = 1
        example['masks'] = array

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(example['masks'], self.random_drop_embedding)
        else:
            image_masks = example['masks']
            text_masks = example['masks']

        example["text_masks"] = text_masks
        example["image_masks"] = image_masks

        if random.uniform(0, 1) < self.prob_use_caption:
            example["caption"] = example["txt"]
        else:
            example["caption"] = ""
        if random.uniform(0, 1) >= self.prob_use_ref:
            example["ref"] = np.zeros((3, 224, 224), dtype=np.float32)

        example['box_ref'] = example['boxes'][0:1]
        example["image"] = example["jpg"].transpose(2, 0, 1)

        example = convert_np_to_tensor(example)
        return example