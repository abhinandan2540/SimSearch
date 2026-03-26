
# constrastive learning required a projection head
# for proper model training

import torch.nn as nn
from .encoder import SimSearchEncoder


class SimSearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimSearchEncoder()

        # self.projector=nn.Sequential(
        #     nn.Linear(256,256), # 256 embedding vector representation
        #     nn.ReLU(), # activation function for contrastive learning to get better representation
        #     nn.Linear(256,256) # output

        # )

        # upgrade for stabilized training, prevent representation collaspe, improving feature quality
        self.projector = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        features = self.encoder(x)  # from img -> feature representations
        # feature representations -> embed projections
        projections = self.projector(features)
        return projections


