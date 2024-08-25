import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, label_dim, feature_d, dropout_p=0.5):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.conv_block(img_channels + label_dim, feature_d, kernel_size=3, stride=2, padding=1, batch_norm=False),
            self.conv_block(feature_d, feature_d * 2, kernel_size=3, stride=2, padding=1),
            self.conv_block(feature_d * 2, feature_d * 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(feature_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout2d(self.dropout_p))
        
        return nn.Sequential(*layers)

    def forward(self, img, label):
        label_embedding = self.label_emb(label).view(label.size(0), -1, 1, 1)
        d_input = torch.cat((img, label_embedding.expand(img.size(0), -1, img.size(2), img.size(3))), dim=1)
        return self.net(d_input)
