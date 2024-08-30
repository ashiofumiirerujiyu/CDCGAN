import torch.nn as nn


class GANGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim, feature=128, dropout_p=0.5):
        super(GANGenerator, self).__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.linear_block(noise_dim, feature),
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
    
    def forward(self, noise):
        return self.net(noise)
    