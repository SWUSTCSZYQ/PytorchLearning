from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MnistDataLoader:

    def __init__(self):

        self.data_train = MNIST('./data', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]))

        self.data_test = MNIST('./data', train=False, download=True, transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]))

        self.data = {}

    def load_data(self):
        date_train_loader = DataLoader(self.data_train, batch_size=256, shuffle=True, num_workers=0)
        date_test_loader = DataLoader(self.data_test, batch_size=256, shuffle=True, num_workers=0)

        self.data['train'] = date_train_loader
        self.data['test'] = date_test_loader
        return self.data
