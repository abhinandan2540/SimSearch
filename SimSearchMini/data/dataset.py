
# what SSL framework like SimCLR does that
# it takes an img and make it two and then works the similarity search into it
# cosine similarity happens

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class SimSearchMiniDataset(Dataset):

    # SimSearchMiniDataset will generate dataset like
    # for a given img -> generate two view of the image (this helps to get cosine similarity)
    # for getting access of image, we need to pass root_dir of image and transformation that we need to apply
    def __init__(self, root_dir, transform):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset) # length of the whole dataset

    def __getitem__(self, index):
        image, _ = self.dataset[index]  # loading the image from the dataset
        # creating two views of the same image
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2
