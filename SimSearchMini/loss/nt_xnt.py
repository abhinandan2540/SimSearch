
# loss function for this model will be nt xnt loss
# nt xnt meaning normalized temperature-scaled cross entropy loss

# it works on the principle of cosine similarity
# same images will have similar embeddings
# different images will have different embeddings


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

