import torch
import torch.nn as nn
from pytorch.CIFAR10.data import Cifar10DataLoader
from pytorch.CIFAR10.model import MyModel


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = Cifar10DataLoader()
    test_data = data_loader.load_data()['test']
    model_path = './CIFAR10_net.pth'
    save_info = torch.load(model_path)
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(save_info)
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(test_data), 'Loss: %.3f | Acc: %.3f%%(%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == '__main__':
    main()