import torch
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_device():
    """
    Get available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def save_model(model, path, optimizer=None, epoch=None, history=None):
    """
    Save model checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'history': history
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {path}")
    return checkpoint.get('epoch', 0), checkpoint.get('history', None)

def count_parameters(model):
    """
    Count trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)