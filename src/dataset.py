import torch.utils.data
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def load_cifar10_data(batch_size):
    """
   This function is designed to retrieve the CIFAR10 dataset from the PyTorch library, apply several transformations
   to it, and ultimately provide data loaders for both the training and testing sets.
    :param batch_size: Batch size refers to the number of samples or data points that are processed simultaneously
                       or at a given time during the training or inference process.
    :return: Data loader object for both the training and testing sets    """

    train_transforms = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))  # keep same as train
    ])

    dataset = CIFAR10(root='data/', download=True, transform=train_transforms)
    test_dataset = CIFAR10(root='data/', train=False, transform=test_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    return train_loader, test_loader