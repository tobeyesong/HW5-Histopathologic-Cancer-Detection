# src/model.py
import timm
import torch.nn as nn
import torch

def get_model(model_name="efficientnet_b0", pretrained=True, num_classes=1):
    """
    Create a model using the timm library.
    
    Args:
        model_name: Name of the model architecture from timm
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        A PyTorch model
    """
    model = timm.create_model(model_name,
                              pretrained=pretrained,
                              num_classes=num_classes)  # binary logits
    return model

def count_params(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    Get optimizer for the model with appropriate parameters
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay factor
        
    Returns:
        PyTorch optimizer
    """
    return torch.optim.AdamW(model.parameters(), 
                            lr=lr, 
                            weight_decay=weight_decay)

if __name__ == "__main__":
    m = get_model()
    print("Trainable params:", count_params(m)/1e6, "M")