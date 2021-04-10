from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Cifar10DataLoader:

    def __init__(self):
        self.data_train = CIFAR10('./data', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]))

        self.data_test = CIFAR10('./data', train=False, download=True, transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.data = {}

    def load_data(self):
        date_train_loader = DataLoader(self.data_train, batch_size=128, shuffle=True, num_workers=0)
        date_test_loader = DataLoader(self.data_test, batch_size=128, shuffle=True, num_workers=0)
        print(len(date_train_loader))
        self.data['train'] = date_train_loader
        self.data['test'] = date_test_loader
        self.data['class'] = self.classes

        return self.data

