import torch
import torch.nn as nn
from pytorch.MNIST.model import LeNet
from pytorch.MNIST.data import MnistDataLoader


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    model.train()  # 切换到训练状态
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    epoch = 20
    dataLoader = MnistDataLoader()
    train_data = dataLoader.load_data()['train']
    for i in range(epoch):
        print('epoch:', i + 1)
        for batch_idx, (inputs, targets) in enumerate(train_data):
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
    path = './mnist_net.pth'  # 保存模型
    torch.save(model.state_dict(), path)
