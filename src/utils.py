import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
train_acc = []
test_acc = []


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


def plot_dataset_images(train_loader, no_images):
    """
    This will plot 'n' (no_images) images for given dataset
    :param train_loader: dataset
    :param no_images: number of images to plot
    :return:
    """
    import math

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(no_images):
        ax = fig.add_subplot(2, math.ceil(no_images / 2), idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])


def get_correct_pred_count(prediction, labels):
    """
     It calculates the count of predictions where the index of the maximum predicted value matches the label index.
     The result is returned as an integer.
    :param prediction: predicted values
    :param labels: corresponding labels
    :return: returns the count of correct predictions
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    """
    This function is responsible for training the model using the provided training data
    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :return:
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test(model, device, test_loader, criterion):
    """
    This function is responsible for testing the trained model using the provided test data.
    :param model:
    :param device:
    :param test_loader:
    :param criterion:
    :return:
    """
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += get_correct_pred_count(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def plot_train_test_accuracy_loss():
    """
    This function is used to plot the training and testing accuracy as well as the training and testing loss.
    It creates a 2x2 grid of subplots in a figure to visualize the four plots.
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def reset_variables():
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
