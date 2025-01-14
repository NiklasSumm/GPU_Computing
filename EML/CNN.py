from __future__ import print_function
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# TODO: Implement the CNN class, as defined in the exercise!
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, 1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 64, 3, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128 , 3, 1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.relu3 = nn.ReLU()
        self.linear0 = nn.Linear(18432, 128)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv0(x)
      x = self.relu0(x)
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.conv2(x)
      x = self.relu2(x)
      x = self.flatten(x)
      x = self.relu3(x)
      x = self.linear0(x)
      x = self.relu4(x)
      x = self.linear1(x)
      x = F.log_softmax(x, dim=1)
      return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print("Using cuda device!")
    else:
        print ("Not using cuda device!")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                    transform=transform)
    dataset_test = datasets.CIFAR10('../data', train=False,
                    transform=transform)
    input_size = 32*32*3
    print("Using CIFAR-10 dataset")

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = CNN().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    start = time.time()
    data_time = []
    data_epoch = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        current_time = time.time() - start
        data_time.append({"Time": current_time, "Accuracy": accuracy})
        data_epoch.append({"Epoch": epoch, "Accuracy": accuracy})


    with open(args.model + "_" + args.dataset + "_" + "lr" + str(args.lr) + "_" + args.optimizer + "_time.csv", 'w', newline='') as csvfile:
        fieldnames = ["Time", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_time)

    with open(args.model + "_" + args.dataset + "_" + "lr" + str(args.lr) + "_" + args.optimizer + "_epoch.csv", 'w', newline='') as csvfile:
        fieldnames = ["Epoch", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_epoch)


if __name__ == '__main__':
    main()