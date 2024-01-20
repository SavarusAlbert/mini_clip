import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()

    def forward(self, image, text):

        return image_features, text_features

