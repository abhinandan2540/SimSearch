
# constrastive learning required a projection head 
# for proper model training

import torch.nn as nn 
from .encoder import SimSearchEncoder

class SimSearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=SimSearchEncoder()
        self.projector=nn.Sequential(
            nn.Linear(128,128), # 128 embedding vector representation
            nn.ReLU(), # activation function for contrastive learning to get better representation
            nn.Linear(128,128) # output

        )

    def forward(self,x):
        features=self.encoder(x) # from img -> feature representations
        projections=self.projector(features) # feature representations -> embed projections
        return projections
    


    