import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from typing import Callable, Dict, List
import requests
import os
from tqdm import tqdm
import timm
from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(encoder_def.output_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = self.fc1(x)
        x = self.dropout(x) if train else x
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class NWayClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, n_way: int = 3):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        
        self.fc1 = nn.Linear(encoder_def.output_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_way)
    
    def forward(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = self.fc1(x)
        x = self.dropout(x) if train else x
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def create_classifier(
    sample: Dict,
    image_keys: List[str],
    n_way: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    image_size: tuple = (128, 128),
):
    shared_backbone = timm.create_model(
        'resnet18',
        pretrained=True,
        num_classes=0,
        global_pool='',
    )
    for param in shared_backbone.parameters():
        param.requires_grad = False
    shared_backbone.eval()
    
    # Get backbone parameter count
    backbone_params = sum(p.numel() for p in shared_backbone.parameters())
    print(f"Using ResNet-18 pretrained on ImageNet with {backbone_params/1e6:.1f}M parameters")
    
    # Create encoders for each image key, sharing the same backbone
    encoders = {}
    for image_key in image_keys:
        encoders[image_key] = PreTrainedResNetEncoder(
            model_name='resnet18',
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            freeze_backbone=True,
            pretrained=True,
            image_size=image_size,
            shared_backbone=shared_backbone,  # Share the frozen backbone
        )
    
    # Create EncodingWrapper (matching your implementation)
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )

    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)

    classifier = classifier_def.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    
    return classifier, optimizer

def load_classifier_func(
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Callable[[Dict], torch.Tensor]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier, _ = create_classifier(sample, image_keys, n_way=n_way, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    @torch.no_grad()
    def predict(obs):
        if isinstance(obs, dict):
            # Convert dict values to tensors if needed
            obs = {k: torch.tensor(v).to(device) if not isinstance(v, torch.Tensor) 
                   else v.to(device) for k, v in obs.items()}
        return classifier(obs, train=False)
    return predict