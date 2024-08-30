import torch.nn as nn


class GANDiscriminator(nn.Module):
    def __init__(self, img_dim, feature=128, dropout_p=0.5):
        super(GANDiscriminator, self).__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.linear_block(img_dim, feature * 4, batch_norm=False),
            self.linear_block(feature * 4, feature * 2),
            self.linear_block(feature * 2, feature),
            nn.Linear(feature, 1),
            nn.Sigmoid(),
        )

    def linear_block(self, in_features, out_features, batch_norm=True):
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(self.dropout_p))

        return nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img)
    