import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
# from .data_utils import * 
# from .dreambooth import BaseDataset_t2i
from pycocotools import mask as mask_utils
from lvis import LVIS
from pycocotools.coco import COCO
# import random
#2.5 version, only add text
import torchvision
from torchvision import transforms

# from .base import BaseDataset
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
import matplotlib.pyplot as plt
import random
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

from PIL import Image, ImageDraw

def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline =colors[bid % len(colors)], width=4)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

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
            A.Rotate(limit=30, p=0.5),
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
        ### test用这个
        while(True):
            try:
                item = self.get_sample(idx)
                return item
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                idx += 1  # 9.10 去了这一句？
                if idx >= len(self.data):
                    raise IndexError("End of dataset reached without finding a valid image")
        

        #             return 0
        #         return self.get_sample(idx)


        # except Exception as e:
        #     try:
        #         print(f"Skipping index {idx} due to error: {e}")
        #         idx = (idx + 1) % len(self.data)
        #     except Exception as e:
        #         try:
        #             print(f"Skipping index {idx+1} due to error: {e}")
        #             idx = (idx + 2) % len(self.data)
        #         except Exception as e:
        #             print(f"Skipping index {idx+2} due to error: {e}")
        #             idx = (idx + 3) % len(self.data)
        #     return self.get_sample(idx) # 0
        
        # item = self.get_sample(idx)
        # return item
        # try:
        #     item = self.get_sample(idx)
        #     return item
        # except:
        #     item = self.get_sample(0)
        #     return item
            # print(idx)
            # return None
            # item = self.get_sample(0)
            # return item
            # # idx = np.random.randint(0, len(self.data)-1)
            # print(idx)
        #### train用这个    
        # while(True):
        #     try:
        #         # idx = np.random.randint(0, len(self.data)-1)
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
    
    def process_boxes(self,ref_image, anno_list, max_ratio = 0.8):
        boxes = []
        layout_list = []
        # layout_multi = torch.zeros(512, 512, 3)
        fill_value = torch.tensor([150/255, 150/255, 150/255], dtype=torch.float32)
        layout_multi = torch.ones((512, 512, 3), dtype=torch.float32) * fill_value
        for anno in anno_list:
            # anno = anno_list[idx]
            ref_mask = self.coco_api.annToMask(anno)
            tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
            # assert mask_score(ref_mask) > 0.90   # 0.90 改低一点，增加训练数量
            # assert self.check_mask_area(ref_mask) == True
            # assert self.check_mask_area(tar_mask)  == True
            # if self.check_mask_area(ref_mask) == False:
            #     continue
            ref_box_yyxx = get_bbox_from_mask(ref_mask)
            # print('ref_box_yyxx',ref_box_yyxx)
            # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.05, mode = 'min') == True
            # Filtering background for the reference image
            ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
            masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑
            y1,y2,x1,x2 = ref_box_yyxx

            ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
            # tar_box_yyxx_crop = box2squre(tar_image, ref_box_yyxx)
            ref_image_squared = pad_to_square(ref_image, pad_value = 0)  #  7.25修改
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
            # layout = np.full((512, 512, 3), (150/255, 150/255, 150/255), dtype=np.float32)
            # layout[y1_new:y2_new,x1_new:x2_new,:] = [1.0, 0.0, 0.0] # 红色

            fill_value = torch.tensor([150/255, 150/255, 150/255], dtype=torch.float32)
            layout1 = torch.ones((512, 512, 3), dtype=torch.float32) * fill_value
            layout = self.draw_bboxes_all(layout1, bbox)
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
        return boxes, layout_all, layout
        
    def process_pairs_customized(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        # assert mask_score(ref_mask) > 0.50   # 0.90 改低一点，增加训练数量
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        # print('ref_box_yyxx',ref_box_yyxx)
        # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # plt.imsave('./random_image.png', ref_mask_3)
        
        masked_ref_image = ref_image * ref_mask_3 + np.zeros_like(ref_image) * 255 * (1-ref_mask_3) #(428, 640, 3) masked_ref_image.png   # 注意： ones 改zeros, 背景为黑

        y1,y2,x1,x2 = ref_box_yyxx

        ####### 6.14 换一种更简单的写法，只做原图级别的剪裁，对于其他也保持一致，保证是整个图
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

        if self.mode == 'train':
            ratio = np.random.randint(11, 15) / 10 
            masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        else:
            pass
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
        
        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        collage = np.zeros(cropped_target_image.shape, dtype=np.uint8)  #改成ones？
 

        collage[y1:y2,x1:x2,:] = ref_image_collage
        
        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2,x1:x2,:] = 1.0

        # if np.random.uniform(0, 1) < 0.7: 
        #     cropped_tar_mask = perturb_mask(cropped_tar_mask)
        #     collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

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
                # layout = layout.copy(),
                # tar_box_yyxx_crop=np.array(tar_box_yyxx_crop), 
                # boxes = boxes.copy()
                ) 
        return item

class COCODataset(BaseDataset_t2i):
    def __init__(self, image_dir, json_path, mode ='train'):
        self.image_dir = image_dir
        self.json_path = json_path
        coco_api = COCO(json_path)
        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        # # anns = [coco_api.getAnnIds[img_id] for img_id in img_ids]
        # for img_id in img_ids:
        #     ann_ids = coco_api.getAnnIds(imgIds=img_id, iscrowd=None) 
        #     anns = coco_api.loadAnns(ann_ids)
            
            

        coco = COCO(json_path)
        ann_ids = coco.getAnnIds()
        # 加载所有的注释
        anns = coco.loadAnns(ann_ids)

        # 将注释按照图像ID分组
        ann_dict = {}
        for ann in anns:
            img_id = ann['image_id']
            if img_id not in ann_dict:
                ann_dict[img_id] = []
            ann_dict[img_id].append(ann)
        
        # 获取所有图像的ID列表
        img_ids = coco.getImgIds()

        # 创建一个字典来存储图像信息和数据
        images = {}

        # 加载所有的图像
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            images[img_id] = {
                'info': img_info,
                'image': image
            }

        self.data = imgs #images
        self.annos = ann_dict #anns
        self.coco_api = coco_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0
        
        # self.layout_path = 'data/coco/images/coco_bbox_train0'
        # self.box_json_path = 'data/coco/bbox_train.json'
        
        # self.ref_dir = 'data/lvis_v1/train_transfered_max'
        # self.ref_dir_max = 'data/lvis_v1/train_ref_max'
        
        self.max_boxes = 30
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])
        # self.random_drop_embedding = 'none'
        self.mode = mode
        if self.mode == 'train':
            self.random_drop_embedding = 'both'
            self.prob_use_caption = 0.9
            self.prob_use_ref = 0.9
            self.coco_json_path = 'data/coco/annotations/captions_train2017.json'
            
        else:
            self.random_drop_embedding = 'none'
            self.prob_use_caption = 1.0
            self.prob_use_ref = 1.0
            self.coco_json_path = 'data/coco/annotations/captions_val2017.json'

        self.coco_captions = self.load_captions(self.coco_json_path)
        self.coco_instances = coco
        self.img_ids = self.coco_instances.getImgIds()
        
    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data
    # 加load coco caption
    def load_captions(self,captions_file):

        with open(captions_file, 'r') as file:
            data = json.load(file)
        captions_dict = {}
        for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
            image_id = item['image_id']
            caption = item['caption']
            if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
                captions_dict[image_id] = caption
        return captions_dict
    
    def get_sample(self, idx):
        # ==== get pairs =====
        # image_name = self.data[idx]['coco_url'].split('/')[-1]
        # image_path = os.path.join(self.image_dir, image_name)
        
        img_id = self.img_ids[idx]
        img_info = self.coco_instances.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        
        # ref_path = os.path.join(self.ref_dir_max,image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_image = image
        
        # anno = self.annos['image_id'] #self.annos[idx]
        obj_ids = []
        # area_tmp = 0
        
        ann_ids = self.coco_instances.getAnnIds(imgIds=img_id)  # ,iscrowd = None导致有的不加载？
        anno = self.coco_instances.loadAnns(ann_ids)
        
        # layout_all = []
        # layout_multi = torch.zeros(image.shape[0], image.shape[1], 3)
        # areas = []
        # if len(anno) == 0:
        #     # return 0
        #     raise ValueError(f"No annotations found for image at index {idx}.")
        # assert len(anno) > 0
        anno = sorted(anno, key=lambda x: x['area'], reverse=True)
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']   
            if area > 0:  # 3600改400
                obj_ids.append(i)
        # assert len(anno) > 0
        
            
        # obj_id = np.random.choice(obj_ids)
        # obj_id = obj_ids[-1]
        # areas = sorted(areas, reverse=True)
        
        # obj_ids.reverse()
        # obj_ids_list = list[reversed(obj_ids)] #obj_ids.reverse()
        anno_list = []
        for obj_id in obj_ids:
            anno_list.append(anno[obj_id])
        # if len(anno) = 0:
        #     anno = []
        #     anno_list = []
        # anno = anno[obj_id]
        anno = anno_list[0]
        ref_mask = self.coco_api.annToMask(anno)  #这个是ref condition
        tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
        
        item_with_collage = self.process_pairs_customized(ref_image, ref_mask, tar_image, tar_mask)  # add layout    #(500,333,3)
        
        # boxes = []
        # for annotation in anno_list:
        #     boxes.append(self.process_boxes(ref_image, ref_mask, tar_image, tar_mask)) 
        #     item_with_collage['boxes'] = torch.cat(boxes, dim=0)
        item_with_collage['boxes'],item_with_collage['layout_all'], item_with_collage['layout'] = self.process_boxes(ref_image, anno_list, max_ratio = 1.0)    
            
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['anno_id'] = image_path
        
        # y1,y2,x1,x2 = item_with_collage['tar_box_yyxx_crop']
        
        # item_with_collage['layout_all'] = item_with_collage['layout']  #这里要改成一一对应的关系
        
        image_crop = self.preprocess(item_with_collage['ref']).float()
        item_with_collage['ref_processed'] = image_crop
        item_with_collage['ref'] = item_with_collage['ref'].transpose(2,0,1).astype(np.float32)
        
        # captions = load_captions(self.coco_json_path)
        annos = self.coco_captions[anno['image_id']]
        item_with_collage['txt'] = annos
        
        category_id = anno['category_id']
        category_info = self.coco_api.loadCats([category_id])[0]
        category_name = category_info['name']
        item_with_collage['positive'] = category_name
        
        item_with_collage['image_id'] = anno['image_id']
        item_with_collage['category_id'] = anno['category_id']
        
        item_with_collage['positive_all'] = []
        prompt_list = []
        # array = np.ones(self.max_boxes,dtype=np.int32) # 从zeros改成ones，不mask了
        array = torch.zeros(self.max_boxes)
        for anno in anno_list:
            category_id = anno['category_id']  #已经按照面积降序排列了
            category_info = self.coco_api.loadCats([category_id])[0]
            prompt_list.append(category_info['name'])

        if len(prompt_list) < self.max_boxes:
            # 补充空字符串至长度为10
            prompt_list.extend([''] * (self.max_boxes - len(prompt_list)))
            item_with_collage['positive_all'] = ','.join(prompt_list)
            array[:len(item_with_collage['positive_all'])] = 1
            item_with_collage['masks'] = array #.reshape(30,1)
        elif len(prompt_list) >= self.max_boxes:
            # 取前10个元素
            prompt_list = prompt_list[:self.max_boxes]
            item_with_collage['positive_all'] = ','.join(prompt_list)
            array[:self.max_boxes] = 1
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
        # item_with_collage["caption"] = item_with_collage['txt'] 
        item_with_collage['box_ref'] = item_with_collage['boxes'][0:1]  #.unsqueeze(1)
        item_with_collage["image"] = item_with_collage["jpg"].transpose(2,0,1) #.transpose(1,2,0)
        
        return item_with_collage

    def __len__(self):
        return len(self.data)

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
