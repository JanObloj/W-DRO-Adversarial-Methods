import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
BATCH_SIZE = 128


class Data:
    def __init__(self, trainset, testset) -> None:
        """
        trainset, testset should be two Datasets
        """
        self.trainloader = DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
        )
        self.testloader = DataLoader(
            testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
        )
        self.X_train, self.y_train = self.splitSet(self.trainloader)
        self.X_test, self.y_test = self.splitSet(self.testloader)

    def splitSet(self, dataloader):
        """
        Return (X, y) as a whole
        """
        X_list, y_list = zip(*((x, y) for x, y in dataloader))
        X_tensor = torch.cat(X_list).cpu()
        y_tensor = torch.cat(y_list).cpu()
        return X_tensor, y_tensor

    def getSplit(self, train=False):
        if train:
            return self.X_train.clone().cpu(), self.y_train.clone().cpu()
        else:
            return self.X_test.clone().cpu(), self.y_test.clone().cpu()

    def getLoader(self, train=False):
        if train:
            return self.trainloader
        else:
            return self.testloader


testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

cifar10 = Data(trainset, testset)


def get_cifar10_split(train=False):
    return cifar10.getSplit(train=train)


def get_cifar10_loader(train=False):
    return cifar10.getLoader(train=train)
