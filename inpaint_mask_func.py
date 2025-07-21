import re
import cv2
import random
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import torch
import torchvision





def draw_masks_from_boxes(boxes, size, randomize_fg_mask=False, random_add_bg_mask=False):
    image_masks = [] 
    for box in boxes:
        image_mask = torch.ones(size,size)
        for bx in box:
            x0,y0,x1,y1 = bx*size
            x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
            obj_width = x1-x0
            obj_height = y1-y0
            if randomize_fg_mask and (random.uniform(0,1)<0.5) and (obj_height>=4) and (obj_width>=4):
                obj_mask = get_a_fg_mask(obj_height, obj_width)
                image_mask[y0:y1,x0:x1] = image_mask[y0:y1,x0:x1] * obj_mask
            else:
                image_mask[y0:y1,x0:x1] = 0

        if random_add_bg_mask and (random.uniform(0,1)<0.5):
            bg_mask = get_a_bg_mask(size)
            image_mask *= bg_mask

        image_masks.append(image_mask)
    return torch.stack(image_masks).unsqueeze(1)





def get_a_fg_mask(height, width):
    assert height>=4 and width>=4 
    size=64
    max_parts=6 
    maxVertex=10
    maxLength=80 
    minBrushWidth=10
    maxBrushWidth=32 
    maxAngle=360
    mask = generate_stroke_mask(im_size=(size,size), 
                                max_parts=max_parts, 
                                maxVertex=maxVertex,
                                maxLength=maxLength,
                                minBrushWidth=minBrushWidth,
                                maxBrushWidth=maxBrushWidth, 
                                maxAngle=maxAngle )
    mask = 1 - torch.tensor(mask)
    
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height, width))
    mask = mask.squeeze(0).squeeze(0)

    return mask  







def get_a_bg_mask(size):
    assert size == 64
    size = 64
    max_parts=4 
    maxVertex=10
    maxLength=32
    maxBrushWidth=12 
    minBrushWidth=3
    maxAngle=360
    mask = generate_stroke_mask( im_size=(size,size), 
                                max_parts=max_parts, 
                                maxVertex=maxVertex,
                                maxLength=maxLength,
                                minBrushWidth=minBrushWidth,
                                maxBrushWidth=maxBrushWidth, 
                                maxAngle=maxAngle )
    mask = 1 - torch.tensor(mask)
    return mask  







# The following code is from BAT-Fill, which is from some other inpainting work I think, maybe Gated Convolution?
# I also made some changes including adding minBrushWidth argument


def generate_stroke_mask(im_size, max_parts=10, maxVertex=20, maxLength=100, minBrushWidth=10, maxBrushWidth=24, maxAngle=360):
    assert minBrushWidth<=maxBrushWidth
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, minBrushWidth, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    # mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask[...,0]

def np_free_form_mask(maxVertex, maxLength, minBrushWidth, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(1,maxVertex + 1)
    startY = np.random.randint(1,h)
    startX = np.random.randint(1,w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(1,maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(minBrushWidth, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask