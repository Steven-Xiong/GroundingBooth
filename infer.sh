# python infer_customized_all.py \
#         --batch_size 1 \
#         --guidance_scale 5 \
#         --folder OUTPUT_test \
#         --dataset dreambench \
#         --ckpt_path checkpoints/checkpoint_00222001.pth

python inference_single.py \
        --batch_size 1 \
        --guidance_scale 3 \
        --folder OUTPUT_test \
        --dataset dreambench \
        --background \
        --ckpt_path checkpoints/checkpoint_00222001.pth