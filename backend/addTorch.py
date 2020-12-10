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
         output += ', '
      output = output[:-1]
   return output

# data functions
def addSum(name, inputs, attributes):
   return '', ('      %s = %s + %s\n') % (name + "_output", inputs['input1'].replace(".", "_"),inputs['input2'].replace(".", "_"))

# loss functions
def addCrossEntropyLoss(attributes):
   return 'criterion = nn.CrossEntropyLoss(' + fillAttributes(attributes) + ')\n'

# optimizers
def addSGD(attributes):
   return 'optimizer = optim.SGD(net.parameters()' + (',' + fillAttributes(attributes) if attributes else '')  + ')\n'

# fully connected layers
def addConv2d(name, inputs, attributes):
   #name = generateName('Conv2d')
   #name = block['name' ])
   # if inputs contains a key named "weights"
   return '      self.' + name + ' = nn.Conv2d(' + fillAttributes(attributes) + ')\n', ('      %s = self.' + name + '(%s)\n') % (name + "_output", inputs['input'].replace(".", "_"))

def addMaxPool2d(name, inputs, attributes):
   #name = generateName('Conv2d')
   #name = block['name' ])
   # if inputs contains a key named "weights"
   return '      self.' + name + ' = nn.MaxPool2d(' + fillAttributes(attributes) + ')\n', ('      %s = self.' + name + '(%s)\n') % (name + "_output", inputs['input'].replace(".", "_"))


def addLinear(name, inputs, attributes):
   #name = generateName('Linear')
   
   return '      self.' + name + ' = nn.Linear(' + fillAttributes(attributes) + ')\n', ('      %s = self.' + name + '(%s)\n') % (name + "_output", inputs['input'].replace('.', '_'))

# activation functions
def addReLU(name, inputs, attributes):
   return '      %s = F.relu(%s)\n' % (name + "_output", inputs['input'].replace('.', "_"))

def addSigmoid(name, inputs, attributes):
   return '      %s = F.sigmoid(%s)\n' % (name + "_output", inputs['input'].replace('.', "_"))

def addTanh(name, inputs, attributes):
   return '      %s = F.tanh(%s)\n' % (name + "_output", inputs['input'].replace('.', "_"))

def addSoftmax(name, inputs, attributes):
   return '      %s = F.softmax(%s, %s)\n' % (name + "_output", inputs['input'].replace('.', "_"), fillAttributes(attributes))
