import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model import get_model
from src.data import get_transforms

torch.backends.cudnn.benchmark = True

# Define TestDS outside the main function so it can be pickled
class TestDS(Dataset):
    def __init__(self, folder, tfm):
        self.files = sorted(Path(folder).glob("*.tif"))
        self.tfm = tfm
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(image=np.array(img))["image"]
        return x, p.stem

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1) load model
    model = get_model().to(device)
    
    # Load model weights
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt    # raw state dict
    model.load_state_dict(state_dict)

    model.eval()

    tfm = get_transforms(train=False)
    ids, preds = [], []

    # 2) batched inference with AMP + progress bar
    test_ds = TestDS(args.img_dir, tfm)
    loader = DataLoader(test_ds,
                        batch_size=64,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    for xb, idb in tqdm(loader, total=len(loader), desc="Inferencing"):
        xb = xb.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            probs = model(xb).sigmoid().cpu().numpy()
        ids.extend(idb)
        # Unwrap the numpy array into Python floats
        for p in probs.reshape(-1):
            preds.append(float(p))

    # 3) write submission
    df = pd.DataFrame({"id": ids, "label": preds})
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(df)} rows)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True,
                   help="folder of test .tif files")
    p.add_argument("--model_path", required=True,
                   help="path to outputs/best.pt")
    p.add_argument("--output", default="submission.csv",
                   help="where to write the CSV")
    args = p.parse_args()
    main(args)