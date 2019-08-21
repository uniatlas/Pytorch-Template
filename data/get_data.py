'''
@Description: Get dataset MNIST CIFAR10 FishionMNIST for offline
@Author: xieydd
@Date: 2019-08-13 17:48:12
@LastEditTime: 2019-08-14 09:33:24
@LastEditors: Please set LastEditors
'''
import torchvision.transforms as transforms
import torchvision
def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_mnist = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        download=True, transform=transform) 
    trainset_cifar10 = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
    trainset_fashion_mnist = torchvision.datasets.FashionMNIST(root='./fashionmnist', train=True,
                                        download=True, transform=transform)

if __name__ == "__main__":
    main()
