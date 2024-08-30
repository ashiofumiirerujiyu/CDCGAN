import torch
import torch.nn as nn


class CGANGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim, feature=128, dropout_p=0.5):
        super(CGANGenerator, self).__init__()
        self.dropout_p = dropout_p
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            self.linear_block(noise_dim + label_dim, feature),
            self.linear_block(feature, feature * 2),
            self.linear_block(feature * 2, feature * 4),
            nn.Linear(feature * 4, img_dim),
            nn.Tanh()
        )

    def linear_block(self, in_features, out_features, batch_norm=True):
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_p))

        return nn.Sequential(*layers)
    
    def forward(self, noise, label):
        embedded_label = self.label_embedding(label)
        print(f"noise: {noise.shape}")
        print(f"label: {label.shape}")
        print(f"embedded_label: {embedded_label.shape}")

        input = torch.cat((noise, embedded_label), dim=1)

        return self.net(input)
    