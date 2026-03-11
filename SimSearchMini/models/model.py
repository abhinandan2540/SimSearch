
# we'll add a projection head into the encoder for maximizing the output embeddings
from .encoder import SimSearchMiniEncoder
import torch.nn as nn 

class SimSearchMiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=SimSearchMiniEncoder() # loads the encoder model

        # a small nn for removing extra noise and other drawbacks 
        # passing the encoded embeddings in input 128 to output 128
        # activation function relu
        self.projector=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):

        # for example input is (32, 3, 224, 224)
        # embedding feature representation will be (32,128), 32 batch size, 128 embedding dimension
        features=self.encoder(x) # returns (32, 128)
        projections=self.projector(features) # feature (encoder embeddings) into projector for better representation
        return projections # returning final embedding representation of size 128
    
    



