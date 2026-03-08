
# for self supervised learning, we'll use the basic concept of the paper
# SimCLR
# it says that
# same images -> similar embeddings
# different images -> different embeddings
import torch
import torch.nn.functional as F


def nt_xnt_loss(z1, z2, temperature=0.5):
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)
    similarity_matrix = similarity_matrix/temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
