from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import csv
import torchvision.ops as tv_nn

class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5, normalization = "none"):
        super().__init__()
        self.layers = self._make_layers(dropout_p, normalization)

    def _make_layers(self, dropout_p, normalization):
        if normalization == "batch":

            layers = [
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(512, 4096),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
            ]
            print("Using batch normalization")
            return nn.ModuleList(layers)
        
        if normalization == "group":
            layers = [
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(16, 128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Dropout(dropout_p),
                nn.GroupNorm(32, 512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(512, 4096),
                nn.Dropout(dropout_p),
                nn.GroupNorm(1, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.Dropout(dropout_p),
                nn.GroupNorm(1, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
            ]
            print("Using group normalization")
            return nn.ModuleList(layers)
        
        layers = [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(4096, 10),
        ]
        print("Using no normalization")
        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output

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
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, epoch):
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

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
    parser.add_argument('--L2_reg', type=float, default=0,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--normalization', type=str, default="none", metavar='N',
                        help='the normalization type')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if (use_cuda):
        print("Using cuda device!")
    else:
        print("Not using cuda device!")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(3)
        ])

    #dataset_train = datasets.SVHN('../data', split='train', download=True, transform=train_transforms)
    #dataset_test = datasets.SVHN('../data', split='test', download=True, transform=test_transforms)

    dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
    dataset_test = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p, normalization=args.normalization).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2_reg)

    start = time.time()
    data_time = []
    data_epoch = []

    print(f'Starting training at: {time.time():.4f}')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader, epoch)
        current_time = time.time() - start
        print("Current time: " + str(current_time))
        data_epoch.append({"Epoch": epoch, "Accuracy": accuracy})
        data_time.append({"Time": current_time, "Accuracy": accuracy})

    with open("Accuracy_time.csv", 'w', newline='') as csvfile:
        fieldnames = ["Time", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_time)

    with open("Accuracy.csv", 'w', newline='') as csvfile:
        fieldnames = ["Epoch", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_epoch)

    torch.set_printoptions(threshold=10_000_000)
    with open("Weights_L2_reg" + str(args.L2_reg) +".txt", "a") as txtfile:
        txtfile.write(str(model.layers[25].weight.data))

    if (args.L2_reg is not None):
        f_name = f'trained_VGG11_L2-{args.L2_reg}.pt'
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')


if __name__ == '__main__':
    main()
