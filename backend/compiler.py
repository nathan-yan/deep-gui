# get pytorch code for fully connected layers
def getFcl(name):
    return 'self.' + name + ' = ' + 'nn.' + name.split('_')[0], 'x = self.' + name + '(x)'

# get pytorch code for functions
def getFunction(name, kind):
    return 'x = ' + kind + '.' + name.split('_')[0] + '(x'

# write the code
def write(orderedNetwork):
    # set up imports
    imports = "\
import torch\n\
import torch.nn as nn\n\
import torch.nn.functional as F\n\
"

    # set up class and constructor
    fcl = "\
\n\
class Net(nn.Module):\n\
    def __init__(self):\n\
        super(Net, self).__init__()\n"
    
    # set up forward function
    forward = "\
    def forward(self, x):\n"
    
    # add pytorch code to the class strings
    while orderedNetwork:
        layer = orderedNetwork.pop(0)
        if layer[2] == 'fcl':
            currentFcl = getFcl(layer[0])
            fcl += "        " + currentFcl[0] + '('
            for parameter in layer[1]:
                fcl += str(parameter) + ','
            fcl = fcl[:-1]
            fcl += ')\n'
            forward += "        " + currentFcl[1] + '\n'
        else:
            forward += "        " + getFunction(layer[0], layer[2])
            for parameter in layer[1]:
                forward += ',' + str(parameter)
            forward += ')\n'
    
    forward += "        return x"

    return imports + fcl + forward

# test input
# 'name': {
    #     'input': '',
    #     'parameters': [],
    #     'kind': '',
    # }
test = {
   "input":{
      "input":"None",
      "parameters":"None",
      "kind":"None"
   },
   "output":{
      "input":"Linear_1",
      "parameters":"None",
      "kind":"None"
   },
   "Linear_1":{
      "input":"relu_1",
      "parameters":[
         100,
         10
      ],
      "kind":"fcl"
   },
   "relu_1":{
      "input":"flatten_1",
      "parameters":[
         
      ],
      "kind":"F"
   },
   "Conv2d_1":{
      "input":"input",
      "parameters":[
         1,
         6,
         3
      ],
      "kind":"fcl"
   },
   "flatten_1":{
      "input":"Conv2d_1",
      "parameters":[
         1
      ],
      "kind":"torch"
   }
}

# build ordered network
def compile(network):
    order = []
    if 'input' in network and 'output' in network:
        layer = network['output']['input']
        while layer != 'input':
            order.append([layer, network[layer]['parameters'], network[layer]['kind']])
            layer = network[layer]['input']
        return order
    else:
        return 'missing input/output'

# check output
print(write(compile(test)))
print(compile(test))