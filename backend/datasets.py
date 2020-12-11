def cifar10Dataset():
    setup = "\
import torchvision\n\
import torchvision.transforms as transforms\n\
\n\
n_mean = 0.5\n\
n_std = 0.5\n\
n_epochs = 2\n\
batch_size = 4\n\
num_workers = 2\n\
\n\
transform = transforms.Compose([\n\
    transforms.ToTensor(),\n\
    transforms.Normalize((n_mean, n_mean, n_mean), (n_std, n_std, n_std))\n\
])\n\n\
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)\n\
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)\n\
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)\n\
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)\n\
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')\n\n"

    evaluation = "\n\
for epoch in range(n_epochs):\n\
    for data in trainloader:\n\
        inputs, labels = data\n\
        optimizer.zero_grad()\n\
        outputs = net(inputs)\n\
        loss = criterion(outputs, labels)\n\
        loss.backward()\n\
        optimizer.step()\n\
\n\
# uncomment to save\n\
# PATH = './cifar_net.pth'\n\
# torch.save(net.state_dict(), PATH)\n\
\n\
correct = 0\n\
total = 0\n\
with torch.no_grad():\n\
    for data in testloader:\n\
        images, labels = data\n\
        outputs = net(images)\n\
        _, predicted = torch.max(outputs.data, 1)\n\
        total += labels.size(0)\n\
        correct += (predicted == labels).sum().item()\n\
\n\
print('Accuracy: ', correct/total)\n\
\n\
class_correct = list(0. for i in range(10))\n\
class_total = list(0. for i in range(10))\n\
with torch.no_grad():\n\
    for data in testloader:\n\
        images, labels = data\n\
        outputs = net(images)\n\
        _, predicted = torch.max(outputs, 1)\n\
        c = (predicted == labels).squeeze()\n\
        for i in range(batch_size):\n\
            label = labels[i]\n\
            class_correct[label] += c[i].item()\n\
            class_total[label] += 1\n\
\n\
for i in range(10):\n\
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))\n"
    
    return setup, evaluation