# Deepfake Detection — Unified Ensemble Framework


### Implementation of the paper:
"A Unified Ensemble Learning Framework for Deepfake Detection
and Robust Spatial Perception: Integrating AI and Digital Forensics"

## Project File Structure

deepfake_project/
│
├── config.py        ← All settings (paths, dims, LRs) — edit this first
├── forensics.py     ← ELA + PRNU (pure numpy, no GPU needed)
├── streams.py       ← All 7 feature extraction backbones (frozen)
├── fusion.py        ← Attention-weighted fusion + meta-classifier (trains)
├── dataset.py       ← Custom dataset with forensics support
├── train.py         ← 4-stage training protocol
├── inference.py     ← Single image prediction with stream breakdown
└── requirements.txt


## STEP 1 — Install dependencies

Open a terminal/Anaconda prompt in your project folder:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm facenet-pytorch Pillow numpy


Verify GPU is detected by :
import torch
print(torch.cuda.is_available())   # must print True if you have GPU
print(torch.cuda.get_device_name(0))

## STEP 2 — Prepare your dataset

Your folder structure must be exactly as folder 0 will represent fake and folder 1 will represent real images:

```
D:\Projects\research\dataset\
│
├── CelebDF\
│   ├── fake\
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── real\
│       ├── image001.jpg
│       └── ...
│
└── WildDeepfake\
    ├── fake\
    │   └── ...
    └── real\
        └── ...
```

**Important:** 
Both datasets must have EXACTLY two subfolders named
`fake` and `real` (lowercase). ImageFolder sorts alphabetically:
fake=0, real=1. If your folders are named differently, rename them.

If your images are raw video frames (not cropped faces), run
face-cropping first using MTCNN (see optional step below).

## STEP 3 — Edit config.py

Open `config.py` and update the paths:

```python
CELEB_PATH = r'D:\Projects\research\dataset\CelebDF'
WILD_PATH  = r'D:\Projects\research\dataset\WildDeepfake'
```

Optionally reduce `BATCH_SIZE` if you run out of VRAM:

```python
BATCH_SIZE = 4 
put batch size as per your memory

## STEP 4 — (Optional) Pre-crop faces with MTCNN

If your images are full video frames, run face-cropping first.
Create a file `crop_faces.py` and run it:

```python
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda')
mtcnn  = MTCNN(image_size=224, margin=40, select_largest=True,
               post_process=False, device=device)

def crop_dataset(src_root, dst_root):
    for label in ['fake', 'real']:
        src_dir = os.path.join(src_root, label)
        dst_dir = os.path.join(dst_root, label)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            try:
                img  = Image.open(src_path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    face_pil = Image.fromarray(
                        (face.permute(1,2,0).numpy() * 255).astype('uint8')
                    )
                    face_pil.save(dst_path)
            except Exception as e:
                print(f'Skipped {fname}: {e}')

crop_dataset(
    r'D:\Projects\research\dataset\CelebDF',
    r'D:\Projects\research\dataset\CelebDF_cropped'
)
crop_dataset(
    r'D:\Projects\research\dataset\WildDeepfake',
    r'D:\Projects\research\dataset\WildDeepfake_cropped'
)
```

Then update `config.py` to point to the `_cropped` folders.

## STEP 5 — Run training


python train.py 
#to train your model 

What happens:
- Stage 1 is already done (all backbones are ImageNet pretrained)
- Stage 2 trains the attention fusion layers (~10 epochs)
- Stage 3 trains the meta-classifier MLP (~10 epochs)
- Stage 4 fine-tunes everything together at low LR (~10 epochs)

Checkpoints saved automatically:
- `weights_stage2_fusion.pth`
- `weights_stage3_meta.pth`
- `weights_final.pth`

Training log saved to: `training_log.csv`


**If you run out of VRAM:**
- Reduce `BATCH_SIZE` to 4 or 2 in config.py
- The 7 backbones are large — 8GB VRAM minimum recommended

## STEP 6 — Run inference (demo)


python inference.py --image path/to/your/image.jpg

Example output:

=======================================================
  VERDICT:    DEEPFAKE
  CONFIDENCE: 91.3%
  Raw P(real):0.0870
=======================================================
  Stream attention weights (higher = more influential):
    Forensics      (ELA+PRNU)           0.187  ████████
    F3-Net         (frequency)          0.163  ███████
    ViT            (global context)     0.148  ██████
    Xception/ResNet (spatial)           0.142  █████
    Swin           (hierarchical)       0.131  █████
    EfficientNet   (spatial)            0.118  █████
    CapsNet        (structural)         0.063  ███
    U-Net encoder  (geometry)           0.048  ██
=======================================================

## STEP 7 — Interpreting results

| Confidence | Meaning |
|---|---|
| >90% DEEPFAKE | High confidence fake — strong multi-stream agreement |
| 70–90% DEEPFAKE | Likely fake — some stream disagreement |
| 50–70% DEEPFAKE | Uncertain — treat with caution |
| <50% either way | Model is not confident — image may be out-of-distribution |
| >70% REAL | Likely genuine |
| >90% REAL | High confidence genuine |

**Which stream matters most:**
- `Forensics` high → compression artifacts detected (ELA fired)
- `F3-Net` high → frequency anomalies (GAN upsampling pattern)
- `ViT/Swin` high → global structural inconsistency detected
- `Xception` high → face boundary blending artifact detected

## VRAM requirements (approximate)

| GPU VRAM | Batch size | Feasible? |
|---|---|---|
| 24 GB | 16 | Comfortable |
| 12 GB | 8 | Recommended |
| 8 GB | 4 | Tight but works |
| 6 GB | 2 | Very slow |
| <6 GB | — | CPU fallback, very slow |

## Common errors and fixes

**If CUDA out of memory:**
```python
# config.py
BATCH_SIZE = 2
```

**Training loss not decreasing:**
- Check your dataset class mapping: `fake=0, real=1` is required
- Ensure images are valid (not corrupt) — add a validation pass
- Try reducing LR in config.py: `LR_STAGE2 = 5e-4`
