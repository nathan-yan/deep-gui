import datasets
import addTorch

# order blocks
def compile(graph):
   orderedBlocks = []
   compiledBlocks = {}
   for block in graph:
      compiledBlocks[block] = False
   for block in graph:
      if not compiledBlocks[block]:
         topologicalSort(graph, block, orderedBlocks, compiledBlocks)
   return orderedBlocks

# recursively stack blocks in order
def topologicalSort(graph, key, stack, visited):
   visited[key] = True
   for value in graph[key]['outputs']:
      if not visited[value]:
         topologicalSort(graph, value, stack, visited)
   stack.append(key)

# write
def write(graph, order):
   # define imports
   imports = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.optim as optim\n\n'

   # define dataset
   dataset = None

   # define network
   net = 'class Net(nn.Module):\n   def __init__(self):\n      super(Net, self).__init__()\n'

   # define forward function
   forward = '\n   def forward(self, x):\n'

   # define initialize model
   initialize = '\nnet = Net()\n'

   # define loss
   loss = None

   # define optimizer
   optimizer = None

   # define train/test
   traintest = None

   # generate script from stack
   while order:
      block = order.pop()
      # check for loss block
      if 'CrossEntropyLoss' == graph[block]['function']:
         loss = addTorch.addCrossEntropyLoss(graph[block]['attributes'])

      # check for optimizer block
      elif 'SGD' == graph[block]['function']:
         optimizer = addTorch.addSGD(graph[block]['attributes'])

      # check for dataset
      elif 'CIFAR10' == graph[block]['function']:
         dataset, traintest = datasets.cifar10Dataset()

      # add blocks to the model
      elif 'Conv2d' == graph[block]['function']:
         addNet, addForward = addTorch.addConv2d(graph[block]['attributes'])
         net += addNet
         forward += addForward
         
      elif 'Linear' == graph[block]['function']:
         addNet, addForward = addTorch.addLinear(graph[block]['attributes'])
         net += addNet
         forward += addForward
      
      elif 'ReLU' == graph[block]['function']:
         forward += addTorch.addReLU()
      
      elif 'Sigmoid' == graph[block]['function']:
         forward += addTorch.addSigmoid()

      elif 'Tanh' == graph[block]['function']:
         forward += addTorch.addReLU()

   return imports + dataset + net + forward + initialize + loss + optimizer + traintest

# test
test = {
   "Conv2d_1":{
      "outputs":[
         "ReLU_1"
      ],
      "attributes":[
         3,
         6,
         5
      ],
      "function":"Conv2d"
   },
   "ReLU_1":{
      "outputs":[
         "Conv2d_2"
      ],
      "attributes":[
         
      ],
      "function":"ReLU"
   },
   "Conv2d_2":{
      "outputs":[
         "ReLU_2"
      ],
      "attributes":[
         6,
         10,
         5
      ],
      "function":"Conv2d"
   },
   "ReLU_2":{
      "outputs":[
         "Flatten_1"
      ],
      "attributes":[
         
      ],
      "function":"ReLU"
   },
   "CrossEntropyLoss_1":{
      "outputs":[
         "SGD_1"
      ],
      "attributes":[
         
      ],
      "function":"CrossEntropyLoss"
   },
   "SGD_1":{
      "outputs":[
         
      ],
      "attributes":[
         "lr=0.001"
      ],
      "function":"SGD"
   },
   "CIFAR10":{
      "outputs":[
         "Conv2d_1"
      ],
      "attributes":[
         
      ],
      "function":"CIFAR10"
   },
   "Flatten_1":{
      "outputs":[
         "Linear_1"
      ],
      "attributes":[
         
      ],
      "function":"Flatten"
   },
   "Linear_1":{
      "outputs":[
         "CrossEntropyLoss_1"
      ],
      "attributes":[
         5760,
         10
      ],
      "function":"Linear"
   }
}

# view generated code
#print(write(test, compile(test)))