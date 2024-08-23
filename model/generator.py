import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim, p=0.5):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, img_dim),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, noise, label):
        label_input = self.label_embedding(label)
        output = self.model(torch.cat((noise, label_input), dim=-1))
        output_img = output.view(output.size(0), 1, 28, 28)

        return output_img
    