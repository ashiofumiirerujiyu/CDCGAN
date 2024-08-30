import torch
import torch.nn as nn


class CGANDiscriminator(nn.Module):
    def __init__(self, img_dim, label_dim, feature=128, dropout_p=0.5):
        super(CGANDiscriminator, self).__init__()
        self.dropout_p = dropout_p
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            self.linear_block(img_dim + label_dim, feature * 4, batch_norm=False),
            self.linear_block(feature * 4, feature * 2),
            self.linear_block(feature * 2, feature),
            nn.Linear(feature, 1),
            nn.Sigmoid()
        )

    def linear_block(self, in_features, out_features, batch_norm=True):
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_p))

        return nn.Sequential(*layers)
    
    def forward(self, img, label):
        embedded_label = self.label_embedding(label)
        input = torch.cat((img, embedded_label), dim=1)

        return self.net(input)
    