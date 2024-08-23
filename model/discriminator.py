import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim, p=0.5):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, label):
        flatten_img = img.view(img.size(0), -1)
        label_input = self.label_embedding(label)
        output = self.model(torch.cat((flatten_img, label_input), dim=-1))

        return output
    