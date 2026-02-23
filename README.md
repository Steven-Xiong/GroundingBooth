# GroundingBooth: Grounding Text-to-Image Customization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![arXiv](https://img.shields.io/badge/arXiv-2409.08520-b31b1b.svg)](https://arxiv.org/abs/2409.08520)
[![TMLR](https://img.shields.io/badge/Accepted-TMLR-blue.svg)](https://jmlr.org/tmlr/)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://groundingbooth.github.io/)

## 👥 Authors

**Zhexiao Xiong**, **Wei Xiong**, **Jing Shi**, **He Zhang**, **Yizhi Song**, **Nathan Jacobs**


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/GroundingBooth.git
cd GroundingBooth

# Create and activate conda environment  
conda env create -f environment.yaml
conda activate groundingbooth
```

## 📥 Model Downloads

### Required Pretrained Models

1. **GroundingBooth Pretrained Model:**
   - Download from: [SharePoint Link](https://gowustl-my.sharepoint.com/:f:/g/personal/x_zhexiao_wustl_edu/Er4Wy-K-u6FAlvOGUAK3NwoBFF8TpIlOcSlA5kjLVDXztA?e=dXFSQO)
   - Place in: `./checkpoints/`

2. **DINOv2 Pretrained Model (ViT-G/14):**
   - Download from: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
   - Place in your model directory

### Directory Structure
```
GroundingBooth/
├── checkpoints/
│   └── checkpoint.pth    # GroundingBooth model
├── configs/                       # Configuration files
├── dataset/                       # Dataset implementations
├── ldm/                          # Latent Diffusion Model components
├── grounding_input/              # Grounding tokenizer inputs
└── dinov2/                       # DINOv2 model components
```

## 🎯 Quick Start

### Basic Inference

Run the default inference pipeline:

```bash
bash infer.sh
```

This executes:
```bash
python inference_single.py \
    --batch_size 1 \
    --guidance_scale 3 \
    --folder OUTPUT_test \
    --dataset dreambench \
    --background \
    --ckpt_path checkpoints/checkpoint.pth
```

### Customized Inference

For specific bounding box control:

```bash
python infer_customized_all.py \
    --batch_size 1 \
    --guidance_scale 5 \
    --folder OUTPUT_custom \
    --dataset dreambench \
    --ckpt_path checkpoints/checkpoint.pth \
    --position 0.1 0.1 0.9 0.9
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--folder` | str | `generation_samples` | Output directory for generated images |
| `--guidance_scale` | float | `3.0` | Classifier-free guidance scale |
| `--dataset` | str | `dreambench` | Dataset type ( `dreambench`) |
| `--ckpt_path` | str | Required | Path to model checkpoint |
| `--position` | float×4 | `(0.25,0.25,0.75,0.75)` | Bounding box coordinates (x1,y1,x2,y2) |
| `--negative_prompt` | str | Auto | Negative prompt for generation |
| `--background` | flag | False | Include background object grounded generation |

## 🏋️ Training

**Note**: Training code will be open-sourced soon. Stay tuned for updates!



## 🔧 Configuration

### Model Configurations

The `configs/` directory contains:
- `inference.yaml`: Inference settings


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## 📚 Citation

If you find GroundingBooth useful in your research, please consider citing:

```bibtex
@article{xiong2024groundingbooth,
  title={Groundingbooth: Grounding text-to-image customization},
  author={Xiong, Zhexiao and Xiong, Wei and Shi, Jing and Zhang, He and Song, Yizhi and Jacobs, Nathan},
  journal={arXiv preprint arXiv:2409.08520},
  year={2024}
}
```

## 📞 Support


- **Email**: [x.zhexiao@wustl.edu](mailto:x.zhexiao@wustl.edu)

---

**Note**: Training code will be released soon. Follow this repository for updates!

