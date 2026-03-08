
# encoder network
# SimSearchEncoder is the model that we need to train to get the embedding representation
# img -> SimSearchEncoder -> embedding vector
import torchvision.models as models
import torch.nn as nn


class SimSearchEncoder(nn.Module):
    def __init__(self, embedding_dim=128): # output embedding dimention 128
        super().__init__()
        # using resnet model for self-supervised learning
        # in the resnet the last layer is for classification on ImageNet classes
        # for self-supervised learning we don't need last layer
        # we want feature representation embeddings
        # resnet: deep residual network for image recognition, resnet 18 has 8 resiudal block
        # 18: 17 convolutional layer and 1 fully connected layer

        resnet = models.resnet18(weights=None)
        # removing the last layer (used in classification purpose)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # resnet outputs 512 feature vector
        # creating 512 dimentional features -> 128 dimentional embeddings
        self.embedding = nn.Linear(512, embedding_dim)  

    def forward(self, x):

        # for example
        # input image in batches as (32,3,224,224) 32 batch size, 3 RGB, 224x224 height, width
        # after passsing throught resnet output as (3,512,1,1)
        x = self.backbone(x)
        x=x.flatten(start_dim=1)
        x=self.embedding(x)
        return x
    
        # x = x.view(x.size(0), -1)  # converts (32,512,1,1) -> (32,512)
        # # projection layer, here all image becomes 128 dimentional embeddings
        # return self.fc(x)

