
# dataset for self-supervised learning
# in most standard SSL frameworks such as SimCLR, BYOL, MoCo, SimSiam
# two augmented views of the same image are used to form a positive pair
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class SimSearchDataset(Dataset):
    def __init__(self, root_dir, transform):
        # root_dir is the directory for images
        self.dataset = ImageFolder(root_dir)
        self.transform = transform  # transformation on the images

    def __len__(self):
        return len(self.dataset)  # length of the total dataset

    def __getitem__(self, index):
        # it only take the images not labels with it
        # self-supervised learning dosen't need labels for training
        # based on embeddings it finds the similarity and dissimilarity
        image, _ = self.dataset[index]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

# each image produces two embedding vectors
