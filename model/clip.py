import torch
import torch.nn as nn
import torch.nn.functional as F
from model.image_encoder import ImageEncoder
from model.text_encoder import TextEncoder


class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class CLIP(nn.Module):
    def __init__(self, temperature=0.5):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = Projection(2048, 256)
        self.text_projection = Projection(768, 256)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # L2正则
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature

        labels = torch.arange(len(batch["input_ids"]), device=logits.device)

        image_loss = F.cross_entropy(logits, labels, axis=0)
        text_loss = F.cross_entropy(logits, labels, axis=1)
        loss = (image_loss + text_loss) / 2
        return loss.mean()