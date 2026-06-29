# CLIP-DGRNet
## CLIP-DGRNet: Dual-Level Dynamic Graph Reasoning with Vision-Language Supervision for Weakly Supervised Semantic Segmentation

- Python 3.6, PyTorch 1.9, and others in environment.yml
- You can create the environment from environment.yml file
- conda env create -f environment.yml

## Usage (PASCAL VOC)

### Step 1. Prepare dataset.
  - Download PASCAL VOC 2012 devkit from official website.
  - You need to specify the path ('voc12_root') of your downloaded devkit in the following steps.

### Step 2. Train ReCAM and generate seeds.
```
  python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_clims_pass True --make_clims_pass True --eval_cam_pass True
```
### Step 3. Train IRN and generate pseudo masks.
```
  python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```
### Step 4. Train semantic segmentation network.
  - To train DeepLab-v2
