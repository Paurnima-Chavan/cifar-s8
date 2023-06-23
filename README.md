# CIFAR-10 Dataset PyTorch implementation
The target is to achieve it in under 50k params with an accuracy of more than 70%
## Basics
The CIFAR-10 dataset is a commonly used benchmark dataset in the field of computer vision and machine learning. It stands for the "Canadian Institute for Advanced Research" and consists of 60,000 small RGB images of size 32x32 pixels. These images are divided into 10 different classes, with each class representing a specific object category.

The dataset is split into two parts: a training set and a test set. The training set contains 50,000 images, while the test set contains 10,000 images. Each image in the dataset is labeled with one of the 10 classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The CIFAR-10 dataset is widely used for various tasks in computer vision, such as image classification, object detection, and image segmentation. It serves as a standard benchmark for evaluating and comparing the performance of different machine learning and deep learning models.

Due to its relatively small image size and the diversity of object classes, the CIFAR-10 dataset provides a challenging yet manageable dataset for developing and testing algorithms in the field of computer vision.
## Data Loader
To begin, the initial step is to load the data and determine the appropriate data transformations.
```bash
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
```
The code defines data transformations for training and testing the CIFAR10 dataset. The training transforms include resizing, horizontal flipping, rotation, affine transformations, color jitter, converting to tensor, and normalization. The test transforms involve converting to tensor and normalizing using predefined mean and standard deviation values.
## Data Visualization
Prior to model creation, it is essential to understand the nature of the data we are working with. To achieve this, we utilize matplotlib to visually inspect the data. By printing a few samples, we can observe and assess their appearance.

![image](https://github.com/Paurnima-Chavan/cifar-s8/assets/25608455/4cbdb9ee-820b-4118-b865-f3dd9e8bffc9)
## Network Architecture
The "BasicBlock" class represents a single block in the Net_1 model. 
It takes an input tensor, performs convolutional operations, applies batch normalization, and applies skip connections to preserve information. 
Each block consists of two convolutional layers (conv1 and conv2) with batch normalization (bn1 and bn2) and ReLU activation functions. 
The skip connection is represented by the shortcut variable, which is a sequence of convolutional and batch normalization layers. The skip connection is activated only if the stride is not equal to 1 or the number of input channels is not equal to the number of output channels.

![image](https://github.com/Paurnima-Chavan/cifar-s8/assets/25608455/8407a443-b82c-4af8-bd85-5447e69db901)

The "Net_1" class represents the overall ResNet model. It initializes with an initial convolutional layer (conv1) and batch normalization (bn1). The model then consists of three stages (layer1, layer2, and layer3), each containing multiple instances of the "BasicBlock" class. The number of channels and stride values increase as we move to deeper stages, following the typical ResNet architecture pattern. After the final stage, the model applies adaptive average pooling (avg_pool) to reduce the spatial dimensions to 1x1. Finally, a fully connected layer (fc) is used to map the flattened feature vector to the output classes.

## Misclassified images

![image](https://github.com/Paurnima-Chavan/cifar-s8/assets/25608455/9f6daf77-b05c-4e45-b59d-b602b38941d5)

## Summary
The provided model is designed for the CIFAR-10 dataset, which has 10 output classes. The number of channels, the number of blocks in each stage, and the stride values are specified in the make_layer method, allowing flexibility in configuring the network architecture.

Overall, this code defines a ResNet model with the "BasicBlock" building blocks, which can be used for image classification tasks on the CIFAR-10 dataset.
