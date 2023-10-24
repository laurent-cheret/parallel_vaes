# data/data_loader.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_mnist_loaders():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers= 4)

    return trainloader, testloader
