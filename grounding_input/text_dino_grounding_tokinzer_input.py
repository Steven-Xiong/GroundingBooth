import os 
import torch as th 
import torch
from omegaconf import OmegaConf
import importlib
# import sys
# sys.path.append('../ldm')
# from util import instantiate_from_config


def instantiate_from_config(config):
    # import pdb; pdb.set_trace()
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class GroundingNetInput:
    def __init__(self):
        self.set = False 
        self.device = torch.device("cuda")
        config = OmegaConf.load('/project/osprey/scratch/x.zhexiao/GLIGEN/configs/coco_text_dino.yaml') 
        # self.image_encoder_global = instantiate_from_config(config.image_encoder_global).cuda() #to(self.device)
    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True
        # import pdb; pdb.set_trace()
        boxes = batch['boxes'].float()
        masks = batch['masks'].float() 
        text_masks = batch['text_masks'].float()
        image_masks = batch['image_masks'].float() 
        text_embeddings = batch["text_embeddings"].float() 
        image_embeddings = batch["image_embeddings"].float()
        # ref_image_emb = self.image_encoder_global.encode( batch["ref"]  ).float() 
        ref_image_emb = batch["ref_image_emb_global"].float()
        ref_box = batch["box_ref"].float()

        self.batch, self.max_box, self.in_dim = text_embeddings.shape
        self.device = 'cuda' #text_embeddings.device
        self.dtype = text_embeddings.dtype
        # ref_image_emb = th.zeros(self.batch, 1, self.in_dim).type(self.dtype).to(self.device) 
        return {"boxes":boxes, 
                "masks":masks, 
                "text_masks":text_masks,
                "image_masks":image_masks,
                "text_embeddings":text_embeddings,
                "image_embeddings":image_embeddings,
                "ref_embeddings": ref_image_emb,
                "ref_box": ref_box,
                }


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        boxes = th.zeros(batch, self.max_box, 4,).type(dtype).to(device) 
        masks = th.zeros(batch, self.max_box).type(dtype).to(device)
        text_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
        image_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
        text_embeddings =  th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device) 
        image_embeddings = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device) 
        ref_embeddings = th.zeros(batch, 1, self.in_dim).type(dtype).to(device) 
        ref_box =  th.zeros(batch, 1, 4).type(dtype).to(device) 

        return {"boxes":boxes, 
                "masks":masks, 
                "text_masks":text_masks,
                "image_masks":image_masks,
                "text_embeddings":text_embeddings,
                "image_embeddings":image_embeddings,
                "ref_embeddings": ref_embeddings,
                "ref_box": ref_box,
                }






