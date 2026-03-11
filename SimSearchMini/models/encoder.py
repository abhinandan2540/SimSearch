
# the model for this consist of two parts
# an encoder and a projection head
# models follow transfer learning approach

import torchvision.models as models
import torch.nn as nn


class SimSearchMiniEncoder(nn.Module):
    def __init__(self, embedding_dim=128):  # taking embedding dimension of 128
        super().__init__()

        resnet = models.resnet18(weights=None)

        # from the resnet architecture, removing the last layer
        # last layer is for classification
        # by removing the last layer, we only have embeddings of images
        # these embeddings helpful in contrastive learning
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # last layer output dimension 512
        self.embeddings = nn.Linear(512, embedding_dim)
        # we are converting 512 -> 128 using nn.Linear

    def forward(self, x):

        # x is image with batch size
        # for example, batch 32, and input image is RGB, x should be (32,3,224,224)
        x = self.backbone(x)  # returns embeddings of size 512 (32, 512,1,1)
        # flattening this
        # (3,512,1,1) -> (32, 512)
        x = x.flatten(start_dim=1)
        x = self.embeddings(x)
        return x  # embedding dimension of (32, 128)
