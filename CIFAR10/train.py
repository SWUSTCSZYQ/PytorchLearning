import torch
import torch.nn as nn
from pytorch.CIFAR10.model import MyModel
from pytorch.CIFAR10.data import Cifar10DataLoader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    model.train()
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    epoch = 40

    data_loader = Cifar10DataLoader()
    train_data = data_loader.load_data()['train']

    for i in range(epoch):
        print('epoch:', i + 1)
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 47 == 46:
                print(batch_idx, len(train_data), 'Loss: %.3f | Acc: %.3f%%(%d/%d)'
                      % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        path = './CIFAR10_net.pth'  # 保存模型
        torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()