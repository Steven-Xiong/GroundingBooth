# GroundingBooth_official




###### pretained model download ######

1. For the pretrained model, please download through this link, and put them into the ./checkpoints folder:

GroundingBooth pretrained model:
https://gowustl-my.sharepoint.com/:f:/g/personal/x_zhexiao_wustl_edu/Er4Wy-K-u6FAlvOGUAK3NwoBFF8TpIlOcSlA5kjLVDXztA?e=dXFSQO

DINOv2 pretrained model:

https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth

we use the vit g/14 model

## test/inference

bash infer.sh

change --ckpt_path to your pretrained model and change --folder to the path you want to save results.

## customized test

For specific box test, use infer_customized_all.py

