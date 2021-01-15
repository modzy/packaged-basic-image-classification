import os
import torch
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, 'model_lib/weights')

resnet_model = models.resnet101(pretrained=True)
torch.save(resnet_model.state_dict(), os.path.join(WEIGHTS_DIR, 'resnet101_weights.pth'))