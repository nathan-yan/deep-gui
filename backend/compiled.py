
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_mean = 0.5
n_std = 0.5
n_epochs = 2
batch_size = 128
num_workers = 0

transform = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
classes = (str(i) for i in range (10))

class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=[3,3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True, padding_mode='zeros',)
      self.maxpool_0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=[1, 1],)
      self.conv2d_1 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=[3,3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True, padding_mode='zeros',)
      self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=[1, 1],)
      self.flatten_0 = nn.Flatten()
      self.dense_0 = nn.Linear(in_features=7*7*20, out_features=10,)

   def forward(self, conv2d_0_input):
      conv2d_0_output = self.conv2d_0(conv2d_0_input)
      relu_0_output = F.relu(conv2d_0_output)
      maxpool_0_output = self.maxpool_0(relu_0_output)
      conv2d_1_output = self.conv2d_1(maxpool_0_output)
      relu_1_output = F.relu(conv2d_1_output)
      maxpool_1_output = self.maxpool_1(relu_1_output)
      flatten_0_output = self.flatten_0(maxpool_1_output)
      relu_2_output = F.relu(flatten_0_output)
      dense_0_output = self.dense_0(relu_2_output)
      softmax_0_output = F.softmax(dense_0_output, dim=1,)

      return softmax_0_output
net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

for epoch in range(n_epochs):
    for data in trainloader:
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)

# uncomment to save
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', correct/total)

class_correct = [0. for i in range(10)]
class_total = [0. for i in range(10)]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    
    