from transformers import CLIPProcessor, CLIPModel

import clip 
import torch
from torchvision import transforms
from PIL import Image

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
        if isinstance(input, list):
            if None in input: return None
        else:
            if input == None: return None
        transform_to_pil = transforms.ToPILImage()
        image = transform_to_pil(input).convert("RGB")
        
        # image = Image.open(input).convert("RGB")
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
        if isinstance(input, list):
            if None in input: return None
        else:
            if input == None: return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


@torch.no_grad()
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
    for phrase, image in zip(phrases, images):
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


@torch.no_grad()
def prepare_batch_hetero(model, processor, metas, max_objs=30):
    batch_boxes = []
    batch_masks = []
    batch_text_masks = []
    batch_image_masks = []
    batch_text_embeddings = []
    batch_image_embeddings = []

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 1024)
    image_embeddings = torch.zeros(max_objs, 1024)
    import pdb; pdb.set_trace()
    ref_features = get_clip_feature(model, processor, metas['ref'],  is_image=True)
    
    
    for meta in metas:
        phrases, images = meta.get("phrases"), meta.get("images")
        images = [None]*len(phrases) if images==None else images 
        phrases = [None]*len(images) if phrases==None else phrases 

        boxes = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
        text_masks = torch.zeros(max_objs)
        image_masks = torch.zeros(max_objs)
        text_embeddings = torch.zeros(max_objs, 768)
        image_embeddings = torch.zeros(max_objs, 768)
    
        text_features = get_clip_feature(model, processor, phrases, is_image=False)
        image_features = get_clip_feature(model, processor, images,  is_image=True)

        n_obj = len(meta['locations'])
        boxes[:n_obj] = torch.tensor(meta['locations'])
        masks[:n_obj] = 1
        if text_features is not None:
            text_embeddings[:n_obj] = text_features
            text_masks[:n_obj] = 1
        if image_features is not None:
            image_embeddings[:n_obj] = image_features
            image_masks[:n_obj] = 1
        
        batch_boxes.append(boxes)
        batch_masks.append(masks)
        batch_text_masks.append(text_masks)
        batch_image_masks.append(image_masks)
        batch_text_embeddings.append(text_embeddings)
        batch_image_embeddings.append(image_embeddings)

    out = {
        "boxes" : torch.stack(batch_boxes),
        "masks" : torch.stack(batch_masks),
        "text_masks" : torch.stack(batch_text_masks)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : torch.stack(batch_image_masks)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : torch.stack(batch_text_embeddings),
        "image_embeddings" : torch.stack(batch_image_embeddings)
    }

    return batch_to_device(out, device) 