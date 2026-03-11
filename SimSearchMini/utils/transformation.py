
# we'll use the same img augmentation techniques of SimCLR
# img augmentations of SimCLR includes
# random crop, horizontal flip, color jitter, gaussian blur, rotation etc

# training a model on normal images is compute extensive, 
# transforming make it easy to do

# taking transformations of SimCLR
import torchvision.transforms as T 

SimSearchMini_transform=T.Compose([
    T.RandomResizedCrop(224), # each image should be size of 224 pixels
    T.RandomHorizontalFlip(), # applyign horizontal flip on image
    T.RandomApply([
        T.ColorJitter(brightness=0.4,contrast=0.4, saturation=0.4, hue=0.1) # applying color transformation on images
    ], p=0.8),
    T.RandomGrayscale(p=0.2), # gray scaling image 
    T.GaussianBlur(kernel_size=23), # blurring image
    T.ToTensor() # img to tensor transformation
    
])

