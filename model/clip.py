import torch
import torch.nn as nn
from model.image_encoder import ImageEncoder
from model.text_encoder import TextEncoder


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        pass

