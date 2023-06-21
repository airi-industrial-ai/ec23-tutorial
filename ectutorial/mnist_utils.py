from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

def get_mnist_0_and_8():
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    zeros = mnist.data[mnist.targets == 0].numpy()
    eights = mnist.data[mnist.targets == 8].numpy()

    rr = torchvision.transforms.RandomResizedCrop((28, 28))
    rotated_zeros = rr(mnist.data[mnist.targets == 0]).numpy()
    rotated_eights = rr(mnist.data[mnist.targets == 8]).numpy()

    return zeros, eights, rotated_zeros, rotated_eights

def gen_sample(zeros, eights):
    tmp = np.concatenate([zeros, eights], axis=0).reshape(-1, 28 ** 2)
    targets = np.concatenate([np.zeros(zeros.shape[0]), np.ones(eights.shape[0])])
    perm = np.random.permutation(targets.shape[0])
    return tmp[perm], targets[perm]

def gen_colored(image):
    channeled = image.reshape(28, 28, 1)
    zeros = np.zeros_like(channeled)
    gs = np.concatenate([channeled] * 3, axis=2)
    red = np.concatenate([channeled] + [zeros] * 2, axis=2)
    green = np.concatenate([zeros, channeled, zeros], axis=2)
    blue = np.concatenate([zeros] * 2 + [channeled], axis=2)
    return gs, red, green, blue

def weighted_grayscale(x, color_index=0):
    weights = [0.3, 0.59, 0.11]
    return x / 255 * weights[color_index]

def vis_colored(image):
    gs, red, green, blue = gen_colored(image / 255)
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(gs, vmin=0, vmax=1)
    axs[1].imshow(red, vmin=0, vmax=1)
    axs[2].imshow(green, vmin=0, vmax=1)
    axs[3].imshow(blue, vmin=0, vmax=1)

def gen_weighted_grayscale(image):
    imgs = [image / 255]
    for c in range(3):
        imgs.append(weighted_grayscale(image, c))
    cimgs = []
    for img in imgs:
        gs, _, _, _ = gen_colored(img)
        cimgs.append(gs)
    return tuple(cimgs)

def vis_greyscale(image):
    gs, red, green, blue = gen_weighted_grayscale(image)
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(gs, vmin=0, vmax=1)
    axs[1].imshow(red, vmin=0, vmax=1)
    axs[2].imshow(green, vmin=0, vmax=1)
    axs[3].imshow(blue, vmin=0, vmax=1)

    
def gen_colored_splits(zeros, eights):
    train_zeros = weighted_grayscale(zeros[:zeros.shape[0] // 4], 0)
    train_eights = weighted_grayscale(eights[:eights.shape[0] // 4], 1)

    train_zeros_flipped = weighted_grayscale(zeros[zeros.shape[0] // 4: zeros.shape[0] // 2], 1)
    train_eights_flipped = weighted_grayscale(eights[eights.shape[0] // 4: eights.shape[0] // 2], 0)

    val_zeros = weighted_grayscale(zeros[zeros.shape[0] // 2: zeros.shape[0] // 4 * 3], 0)
    val_eights = weighted_grayscale(eights[eights.shape[0] // 2: eights.shape[0] // 4 * 3], 1)

    test_zeros = weighted_grayscale(zeros[zeros.shape[0] // 4 * 3:], 1)
    test_eights = weighted_grayscale(eights[eights.shape[0] // 4 * 3:], 0)
    
    blue_zeros = weighted_grayscale(zeros[zeros.shape[0] // 4 * 3:], 2)
    blue_eights = weighted_grayscale(eights[eights.shape[0] // 4 * 3:], 2)

    return {
        "train": gen_sample(train_zeros, train_eights),
        "train_flipped": gen_sample(train_zeros_flipped, train_eights_flipped),
        "val": gen_sample(val_zeros, val_eights),
        "test": gen_sample(test_zeros, test_eights),
        "blue": gen_sample(blue_zeros, blue_eights)
    }