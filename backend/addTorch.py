names = {}

# create new names
def generateName(name):
   newName = name
   if name in names.keys():
      names[name] += 1
   else:
      names[name] = 1
   newName += '_' + str(names[name])
   return newName

# fill in the attributes
def fillAttributes(attributes):
   output = ''
   if attributes:
      for attribute in attributes:
         if '=' in str(attribute):
            output += attribute
         else:
            output += str(attribute)
         output += ','
      output = output[:-1]
   return output

# loss functions
def addCrossEntropyLoss(attributes):
   return 'criterion = nn.CrossEntropyLoss(' + fillAttributes(attributes) + ')\n'

# optimizers
def addSGD(attributes):
   return 'optimizer = optim.SGD(net.parameters()' + (',' + fillAttributes(attributes) if attributes else '')  + ')\n'

# fully connected layers
def addConv2d(attributes):
   name = generateName('Conv2d')
   return '      self.' + name + ' = nn.Conv2d(' + fillAttributes(attributes) + ')\n', '      x = self.' + name + '(x)\n'

def addLinear(attributes):
   name = generateName('Linear')
   return '      self.' + name + ' = nn.Linear(' + fillAttributes(attributes) + ')\n', '      x = self.' + name + '(x)\n'

# activation functions
def addReLU():
   return '      x = F.relu(x)\n'

def addSigmoid():
   return '      x = F.sigmoid(x)\n'

def addTanh():
   return '      x = F.tanh(x)\n'
