
# for self supervised learning, we'll use the basic concept of the paper
# SimCLR
# it says that
# same images -> similar embeddings
# different images -> different embeddings

import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):

    batch_size = z1.shape[0]

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)

    similarity_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size)
    ])

    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size).long().to(z.device)
    loss = F.cross_entropy(logits, labels)
    return loss

