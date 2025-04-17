# src/train.py
import torch
import json
import argparse
import os
import time
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

from src.data import get_loaders
from src.model import get_model, get_optimizer

import torch
torch.backends.cudnn.benchmark = True


def train_one_epoch(model, dl, optim, scaler, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0
    
    for x, y in tqdm(dl, leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        
        with autocast():
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                       logits, y)
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        
        running_loss += loss.item() * x.size(0)
        
    return running_loss / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, device):
    """Evaluate model on validation set"""
    model.eval()
    auroc = BinaryAUROC().to(device)
    val_loss = 0
    
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        val_loss += F.binary_cross_entropy_with_logits(
                        logits, y, reduction='sum').item()
        auroc.update(logits.sigmoid(), y.int())
        
    return val_loss / len(dl.dataset), auroc.compute().item()

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save command line arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Get data loaders
    train_dl, val_dl = get_loaders(args.img_dir,
                                   args.csv_path,
                                   batch=args.batch,
                                   seed=args.seed)
    
    # Create model and optimizer
    model = get_model(model_name=args.model,
                     pretrained=not args.no_pretrained).to(device)
    optim = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    if args.use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, 
                                     patience=2, verbose=True)
    else:
        scheduler = None
    
    # Create mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_auc, history = 0, []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss = train_one_epoch(model, train_dl, optim, scaler, device)
        
        # Evaluate
        val_loss, val_auc = evaluate(model, val_dl, device)
        
        # Step scheduler if used
        if scheduler is not None:
            scheduler.step(val_auc)
        
        # Log metrics
        elapsed = time.time() - start_time
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "elapsed_time": elapsed
        })
        
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "val_auc": val_auc,
                "epoch": epoch,
            }, output_dir / "best.pt")
            print(f"Saved new best model with val_auc={val_auc:.4f}")
            
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "val_auc": val_auc,
        "epoch": epoch,
    }, output_dir / "final.pt")
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True, help="Directory with image files")
    p.add_argument("--csv_path", required=True, help="Path to CSV file with labels")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--model", default="efficientnet_b0", help="Model architecture from timm")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--batch", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_pretrained", action="store_true", help="Don't use pretrained weights")
    p.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    
    args = p.parse_args()
    main(args)