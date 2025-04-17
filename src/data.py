# src/data.py
import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image

# You can calculate these from your dataset for better results
PCAM_MEAN = (0.5, 0.5, 0.5)
PCAM_STD = (0.5, 0.5, 0.5)

# ---------- transforms ----------
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2,
                          contrast=0.2,
                          saturation=0.2,
                          hue=0.05, p=0.5),
            A.Normalize(mean=PCAM_MEAN, std=PCAM_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=PCAM_MEAN, std=PCAM_STD),
            ToTensorV2(),
        ])

# ---------- dataset ----------
class PCam(Dataset):
    def __init__(self, img_dir, csv_path, ids, transforms):
        self.img_dir = Path(img_dir)
        self.labels = pd.read_csv(csv_path,
                                  index_col="id").loc[ids]["label"].values
        self.ids = ids
        self.tfms = transforms

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = self.img_dir / f"{img_id}.tif"
        
        try:
            # Option 1: Use Pillow (more reliable for TIFFs on Windows)
            img = Image.open(path).convert("RGB")
            img = np.array(img)
            
            # Option 2: Use OpenCV (uncomment if preferred)
            # img = cv2.imread(str(path))
            # if img is None:
            #     raise ValueError(f"Failed to load image: {path}")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = self.tfms(image=img)["image"]
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label.unsqueeze(0)  # shape (1,)
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image as fallback (assuming PCam's 96x96 size)
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            img = self.tfms(image=img)["image"]
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label.unsqueeze(0)

# ---------- factory ----------
def get_loaders(img_dir, csv_path,
                batch=256, val_frac=0.1, num_workers=4, seed=42):
    df = pd.read_csv(csv_path)
    ids = df["id"].tolist()
    
    # Optional: shuffle before splitting for better distribution
    if seed is not None:
        import random
        random.seed(seed)
        random.shuffle(ids)
        
    split = int(len(ids)*(1-val_frac))
    train_ids, val_ids = ids[:split], ids[split:]

    train_ds = PCam(img_dir, csv_path, train_ids,
                    get_transforms(train=True))
    val_ds = PCam(img_dir, csv_path, val_ids,
                  get_transforms(train=False))

    train_dl = DataLoader(train_ds, batch_size=batch,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch*2,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
    return train_dl, val_dl