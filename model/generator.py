import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_channels, feature_g, dropout_p=0.5):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.deconv_block(noise_dim + label_dim, feature_g * 8, kernel_size=3, stride=2, padding=0),
            self.deconv_block(feature_g * 8, feature_g * 4, kernel_size=3, stride=2, padding=0),
            self.deconv_block(feature_g * 4, feature_g * 2, kernel_size=3, stride=2, padding=1),
            self.deconv_block(feature_g * 2, feature_g, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(feature_g, img_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(self.dropout_p))
        
        return nn.Sequential(*layers)

    def forward(self, noise, label):
        label_embedding = self.label_emb(label).view(label.size(0), -1, 1, 1)
        noise = noise.view(noise.size(0), -1, 1, 1)
        gen_input = torch.cat((noise, label_embedding), dim=1)
        return self.net(gen_input)
    