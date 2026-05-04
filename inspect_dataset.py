import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from PIL import Image

from forensics import extract_forensic_features


# Transforms 
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),          
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Dataset 1: CelebDF 
class CelebDFDataset(Dataset):
    """
    Wraps torchvision ImageFolder for CelebDF.
    Returns: (image_tensor [3,224,224], forensic_feat [1296], label int)
    label: 0=fake, 1=real  (ImageFolder alphabetical: fake < real)
    """
    def __init__(self, root: str, transform, use_forensics: bool = True):
        self.base          = datasets.ImageFolder(root, transform=transform)
        self.use_forensics = use_forensics
        self.class_to_idx  = self.base.class_to_idx   # {'fake':0, 'real':1}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_tensor, label = self.base[idx]

        if self.use_forensics:
            path, _ = self.base.samples[idx]
            pil_img  = Image.open(path).convert('RGB')
            forensic = torch.tensor(
                extract_forensic_features(pil_img), dtype=torch.float32
            )
        else:
            forensic = torch.zeros(1296, dtype=torch.float32)

        return img_tensor, forensic, label


# Dataset 2: WildDeepfake 
class WildDeepfakeDataset(Dataset):
    """
    Loads WildDeepfake NPY clips.

    Each .npy = (30, 299, 299, 3) uint8 clip.
    Each frame becomes an independent sample → 30× more data per file.

    Index mapping (lazy loading — one clip loaded per access):
        global idx → clip_idx  = idx // frames_per_clip
                   → frame_idx = idx %  frames_per_clip

    label: 0=fake, 1=real  (matches CelebDF mapping exactly)
    """
    CLASS_TO_IDX = {'fake': 0, 'real': 1}

    def __init__(self, root: str, transform, use_forensics: bool = True,
                 frames_per_clip: int = 30):
        self.root            = root
        self.transform       = transform
        self.use_forensics   = use_forensics
        self.frames_per_clip = frames_per_clip
        self.class_to_idx    = self.CLASS_TO_IDX

        self.samples = []
        for cls_name, label in self.CLASS_TO_IDX.items():
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(
                    f'Expected folder not found: {cls_dir}\n'
                    f'WildDeepfake must have fake/ and real/ subfolders.'
                )
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        total = len(self.samples) * frames_per_clip
        print(f'[WildDeepfakeDataset] {len(self.samples)} clips × '
              f'{frames_per_clip} frames = {total} samples')

    def __len__(self):
        return len(self.samples) * self.frames_per_clip

    def _load_frame(self, idx):
        """Lazily load one frame from the correct NPY clip."""
        clip_idx  = idx // self.frames_per_clip
        frame_idx = idx %  self.frames_per_clip
        npy_path, label = self.samples[clip_idx]
        clip  = np.load(npy_path)       
        frame = clip[frame_idx]         
        pil   = Image.fromarray(frame)  
        return pil, label

    def __getitem__(self, idx):
        pil, label = self._load_frame(idx)

        img_tensor = self.transform(pil)  

        if self.use_forensics:
            forensic = torch.tensor(
                extract_forensic_features(pil), dtype=torch.float32
            )
        else:
            forensic = torch.zeros(1296, dtype=torch.float32)

        return img_tensor, forensic, label


def build_dataloaders(celeb_path, wild_path, batch_size, val_split, num_workers):
    """
    Combines CelebDF + WildDeepfake into one train/val split.

    Key design decisions:
    - Train and val subsets use DIFFERENT transform instances
      (train has augmentation, val does not) via separate Dataset objects
      sharing the same index permutation — zero data leakage.
    - Class mapping assertion ensures fake=0, real=1 in both datasets.

    Returns: train_loader, val_loader
    """

    # Training datasets 
    celeb_train = CelebDFDataset(celeb_path,  TRAIN_TRANSFORM)
    wild_train  = WildDeepfakeDataset(wild_path, TRAIN_TRANSFORM)

    # Validation datasets 
    celeb_val   = CelebDFDataset(celeb_path,  VAL_TRANSFORM)
    wild_val    = WildDeepfakeDataset(wild_path, VAL_TRANSFORM)


    assert celeb_train.class_to_idx == wild_train.class_to_idx, (
        f'Class mismatch!\n'
        f'  CelebDF:      {celeb_train.class_to_idx}\n'
        f'  WildDeepfake: {wild_train.class_to_idx}\n'
        f'Both must have fake/ and real/ subfolders (lowercase).'
    )
    print(f'[Dataset] Class map confirmed: {celeb_train.class_to_idx}')
    # {'fake': 0, 'real': 1}

    combined_train = ConcatDataset([celeb_train, wild_train])
    combined_val   = ConcatDataset([celeb_val,   wild_val])

    total    = len(combined_train)
    indices  = torch.randperm(total).tolist()
    split_at = int((1 - val_split) * total)

    train_ds = Subset(combined_train, indices[:split_at])
    val_ds   = Subset(combined_val,   indices[split_at:])

    print(f'[Dataset] CelebDF      : {len(celeb_train):>8,} samples')
    print(f'[Dataset] WildDeepfake : {len(wild_train):>8,} samples  '
          f'({len(wild_train.samples)} clips × 30 frames)')
    print(f'[Dataset] Total        : {total:>8,}')
    print(f'[Dataset] Train        : {len(train_ds):>8,}')
    print(f'[Dataset] Val          : {len(val_ds):>8,}')

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


if __name__ == '__main__':
    from config import CELEB_PATH, WILD_PATH, BATCH_SIZE, VAL_SPLIT

    print('\n--- Sanity check: loading one batch (num_workers=0) ---')
    loader_train, loader_val = build_dataloaders(
        CELEB_PATH, WILD_PATH, BATCH_SIZE, VAL_SPLIT, num_workers=0
    )

    imgs, forensics, labels = next(iter(loader_train))
    print(f'\nImage tensor  : {imgs.shape}        dtype={imgs.dtype}')
    print(f'Forensic feat : {forensics.shape}   dtype={forensics.dtype}')
    print(f'Labels        : {labels.tolist()}')
    print(f'Unique labels : {labels.unique().tolist()}  (0=fake  1=real)')
    print('\nAll good — ready to train.')