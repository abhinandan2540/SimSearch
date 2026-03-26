
# self supervised learning uses two views of the same image
# creating two images using augmentation techniques such as
# random crop, color jitter, flip, blur, rotation etc


# img transformation pre self-supervised learning
import torchvision.transforms as T

# img transform taken from SimCLR
SimSearch_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.0)),  # img size 224 pxls
    T.RandomHorizontalFlip(),  # horizontal flip
    T.RandomApply([
        T.ColorJitter(  # controlling brightness, saturation, heu
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
    ], p=0.8),
    T.RandomGrayscale(p=0.2),  # grayscaling image
    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  # blurring img
    T.RandomRotation(15),
    T.ToTensor()  # transforming img into tensor format
])

# successfull representation mostly depends on how well data is augmented
