# 🎨 ReStyle3D: Scene-Level Appearance Transfer with Semantic Correspondences

### ACM SIGGRAPH 2025

 [![ProjectPage](https://img.shields.io/badge/Project_Page-ReStyle3D-blue)](https://restyle3d.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2502.10377-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2502.10377) [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/gradient-spaces/ReStyle3D) [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

Official implementation of the paper titled "Scene-level Appearance Transfer with Semantic Correspondences".

[Liyuan Zhu](https://www.zhuliyuan.net/)<sup>1</sup>,
[Shengqu Cai](https://primecai.github.io/)<sup>1,\*</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>2,\*</sup>,
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<sup>1</sup>,
[Naji Khosravan](https://www.najikhosravan.com/)<sup>3</sup>,
[Iro Armeni](https://ir0.github.io/)<sup>1</sup>



<sup>1</sup>Stanford University, <sup>2</sup>NVIDIA Research, <sup>3</sup>Zillow Group | <sup>\*</sup> denotes equal contribution


```bibtex
@inproceedings{zhu2025_restyle3d,
    author = {Liyuan Zhu and Shengqu Cai and Shengyu Huang and Gordon Wetzstein and Naji Khosravan and Iro Armeni},
    title = {Scene-level Appearance Transfer with Semantic Correspondences},
    booktitle = {ACM SIGGRAPH 2025 Conference Papers},
    year = {2025},
  }
```

We introduce ReStyle3D, a novel framework for scene-level appearance
transfer from a single style image to a real-world scene represented by
multiple views. This method combines explicit semantic correspondences
with multi-view consistency to achieve precise and coherent stylization.
<p align="center">
  <a href="">
    <img src="https://arxiv.org/html/2502.10377v1/x1.png" width="100%">
  </a>
</p>


## 🛠️ Setup
### ✅ Tested Environments
- Ubuntu 22.04 LTS, Python 3.10.15, CUDA 12.2, GeForce RTX 4090/3090

- CentOS Linux 7, Python 3.12.1, CUDA 12.4, NVIDIA A100

### 📦 Repository
```
git clone git@github.com:GradientSpaces/ReStyle3D.git
```
conda create -n restyle3d python=3.10
conda activate restyle3d
pip install -r requirements.txt
```

### 📦 Pretrained Checkpoints
Download the pretrained models by running:
```
bash scripts/download_weights.sh
```


## 🚀 Usage

We download our dataset:
```
bash scripts/download_data.sh
```

### 🎮 Demo (Single-view)
We include 3 demo images to run semantic appearance transfer:
```
python restyle_image.py
```



### 🎨 Stylizing Multi-view Scenes 
To run on a single scene and style:
```
python restyle_scene.py   \
 --scene_type bedroom   \
 --style_path demo/design_styles/bedroom/pexels-itsterrymag-2631746

### 📂 Dataset: SceneTransfer

1. Interior Scenes:
```
📁 data/
      ├── bedroom/
      │   ├── 0/
      │   │   ├── depth/       # depth maps
      │   │   └── seg_dict/    # semantic segmentation dictionaries
      │       └── ...
      ├── living_room/
      └── kitchen/
```
```
📁 data/
  └── design_styles/
      ├── bedroom/
      │   └── pexels-itsterrymag-2631746/
      │       ├── image.jpg        # style reference image
      │       ├── seg_dict.pth     # semantic segmentation dictionary 
      │       └── seg.png          # segmentation visualization
      ├── living_room/
      └── kitchen/
```





## 🚧 TODO
- [x] Release full dataset
- [ ] Release evaluation code
- [ ] Customize dataset


## 🙏 Acknowledgement
Our codebase is built on top of the following works:
- [Cross-image-attention](https://github.com/garibida/cross-image-attention) 
- [ODISE](https://github.com/NVlabs/ODISE)
- [ViewCrafter](https://github.com/Drexubery/ViewCrafter)
- [GenWarp](https://github.com/sony/genwarp)
- [DUSt3R](https://github.com/naver/dust3r) 

We appreciate the open-source efforts from the authors.

## 📫 Contact
If you encounter any issues or have questions, feel free to reach out: [Liyuan Zhu](liyzhu@stanford.edu).




## Batch CLI Pipeline (Adapted)

The script scripts/batch_canny_depth_control.py now runs a folder-to-folder batch pipeline by delegating each transfer to the same internal flow used by restyle_image.py.

Example with one global style image:
```
python scripts/batch_canny_depth_control.py \
  --prompt "modern interior with warm wood textures" \
  --scale 2.0 \
  --content_dir data/interiors/bedroom/0/images \
  --format .jpg \
  --output_dir output/batch_demo \
  --style_image data/design_styles/bedroom/pexels-itsterrymag-2631746/image.jpg
```

Example with random style per input image:
```
python scripts/batch_canny_depth_control.py \
  --prompt "modern interior with warm wood textures" \
  --scale 2.0 \
  --content_dir data/interiors/bedroom/0/images \
  --format .jpg \
  --output_dir output/batch_demo \
  --style_dir data/design_styles/bedroom
```

Behavior:
- The script iterates over all files in content_dir matching format.
- It saves one stylized image per input file into output_dir using the same filename.
- style_dir overrides style_image and randomly selects one style reference for each input image.
- Required segmentation dictionaries are resolved automatically from common project layouts (seg_dict folders, sibling .pth files, or style folders with image.* plus seg_dict.pth).

## Automatic Hugging Face Checkpoint Loading

When the batch CLI starts, it automatically prefetches all Hugging Face checkpoints used by the ReStyle3D single-view transfer and refinement path.

Prefetched repositories:
1. runwayml/stable-diffusion-v1-5
2. lllyasviel/sd-controlnet-depth
3. diffusers/controlnet-depth-sdxl-1.0
4. madebyollin/sdxl-vae-fp16-fix
5. stabilityai/stable-diffusion-xl-base-1.0

This ensures all required Hugging Face assets are in local cache before processing the first content image.

## Detailed Workflow And Model Stack

The adapted batch script executes this workflow for each input image:

1. Input discovery
- Collect all content images from content_dir matching format.
- Resolve one style image (global style_image, random from style_dir, or self-style fallback).

2. Semantic data resolution
- Resolve structure segmentation dictionary for the content image.
- Resolve style segmentation dictionary for the chosen style image.

3. RunConfig assembly
- Build scene_transfer.config.RunConfig with:
  - prompt
  - swap_guidance_scale (from scale)
  - num_timesteps (steps)
  - skip_steps
  - load_latents
  - domain_name

4. Pipeline initialization and reuse
- restyle_image.ModelManager is initialized once and reused across the loop.
- It creates:
  - transfer_pipe (SceneTransfer SD1.5 semantic transfer)
  - refiner_pipe (SDXL ControlNet refiner)

5. Latent and depth stage
- restyle_image.run invokes scene_transfer.latent_utils.load_latents_or_invert_images.
- If latents are cached, they are reused.
- Otherwise, images are inverted and depth maps are estimated/loaded.

6. Semantic transfer stage (SD1.5)
- Uses SD1.5 plus depth ControlNet and semantic attention to transfer style while preserving scene structure.

7. Refinement stage (SDXL)
- The SDXL base model plus SDXL depth ControlNet and SDXL VAE refine low-resolution transfer into a high-resolution stylized output.

8. Final write
- The script writes the final stylized image to output_dir/<original_input_filename>.
- Intermediate artifacts are stored under output_dir/_restyle3d_work for caching and debugging.

## Where Checkpoints Are Loaded In Code

- scene_transfer/model_utils.py
  - get_scene_transfer_sd15:
    - runwayml/stable-diffusion-v1-5
    - lllyasviel/sd-controlnet-depth
  - get_refining_pipe:
    - stabilityai/stable-diffusion-xl-base-1.0
    - diffusers/controlnet-depth-sdxl-1.0
    - madebyollin/sdxl-vae-fp16-fix

- restyle_image.py
  - ModelManager initializes both transfer and refiner pipelines.
  - run performs inversion, semantic matching, transfer, and refinement.

Additional non-Hugging Face assets:
- scripts/download_weights.sh downloads project-local weights such as Depth Anything V2 and DUSt3R checkpoints.
- scene_transfer/depth_estimator.py expects checkpoints/depth_anything_v2_vitl.pth for the default depth estimator path.
