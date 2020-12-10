import datasets

# test json
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
      "function":"nn.Conv2d"
   },
   "ReLU_1":{
      "outputs":[
         "Conv2d_2"
      ],
      "attributes":[
         
      ],
      "function":"nn.ReLU"
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
      "function":"nn.Conv2d"
   },
   "ReLU_2":{
      "outputs":[
         "Flatten_1"
      ],
      "attributes":[
         
      ],
      "function":"nn.ReLU"
   },
   "CrossEntropyLoss_1":{
      "outputs":[
         "SGD_1"
      ],
      "attributes":[
         
      ],
      "function":"nn.CrossEntropyLoss"
   },
   "SGD_1":{
      "outputs":[
         
      ],
      "attributes":[
         "lr=0.001"
      ],
      "function":"optim.SGD"
   },
   "CIFAR10":{
      "outputs":[
         "Conv2d_1"
      ],
      "attributes":[
         
      ],
      "function":"cifar10_dataset"
   },
   "Flatten_1":{
      "outputs":[
         "Linear_1"
      ],
      "attributes":[
         
      ],
      "function":"nn.Flatten"
   },
   "Linear_1":{
      "outputs":[
         "CrossEntropyLoss_1"
      ],
      "attributes":[
         5760,
         10
      ],
      "function":"nn.Linear"
   }
}

test = {
   "conv2d_1" : {
      "outputs" : [
         "relu_1.input",
         "conv2d_3.input"
      ],
      "inputs" : [],
      "attributes" : [
         3, 6, 5
      ]
   },
   "relu_1" : {
      "outputs" : [
         "conv2d_2.input",
      ],
      "inputs" : {
         "input" : "conv2d_1.output"
      }
   },
   "conv2d_2" : {
      "outputs" : [
         "add.input1"
      ],
      "inputs" : {
         "input" : "relu_1.output"
      }
   },
   "conv2d_3" : {
      "outputs" : [
         "add.input2"
      ],
      "inputs" : {
         "input" : "conv2d_1.output"
      }
   },
   "add": {
      "outputs" : [

      ],
      "inputs" : {
         "input1" : "conv2d_2.output",
         "input2" : "conv2d_3.output"
      }
   }
}

test = {"inp":{"type" : "input_data", "inputs":{},"attributes":["file path=./input_file.g","shuffle=True"]},"conv_1":{"type" : "conv2d", "inputs":{"input":"inp.data"},"attributes":["in_channels=3","out_channels=6","kernel_size=[3,3]","stride=[1,1]","padding=[1,1]","dilation=[1,1]","groups=1","bias=True","padding_mode=zeros"]},"relu_1":{"type" : "relu", "inputs":{"input":"conv_1.output"},"attributes":[]},"conv_2":{"type" : "conv2d", "inputs":{"input":"relu_1.output"},"attributes":["in_channels=6","out_channels=12","kernel_size=[3,3]","stride=[1,1]","padding=[1,1]","dilation=[1,1]","groups=1","bias=True","padding_mode=zeros"]},"relu_2":{"type" : "relu", "inputs":{"input":"conv_2.output"},"attributes":[]},"conv_3":{"type" : "conv", "inputs":{"input":"relu_2.output"},"attributes":["in_channels=12","out_channels=24","kernel_size=[3,3]","stride=[1,1]","padding=[1,1]","dilation=[1,1]","groups=1","bias=True","padding_mode=zeros"]},"add_1":{"type" : "add", "inputs":{"input2":"conv_1.output","input1":"conv_3.output"},"attributes":[]},"xent_1":{"type" : "cross_entropy", "inputs":{"prediction":"add_1.output","target":"inp.target"},"attributes":[]}}

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
    for value in graph[key]['inputs']:
        if not visited[graph[key]['inputs'][value].split(".")[0]]:
            topologicalSort(graph, value, stack, visited)
    stack.append(key)

# write
def write(graph, order):
    # define imports
    imports = 'import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n'

    # define dataset
    dataset = None

    # define sequential model
    net = """class Network(nn.Module):
      def __init__(self):
         super(Network, self).__init__()"""

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
        if 'Loss' in graph[block]['function']:
            loss = 'criterion = ' + graph[block]['function'] + '(' + fillAttributes(graph, block) + ')\n'

        # check for optimizer block
        elif 'optim' in graph[block]['function']:
            optimizer = 'optimizer = ' + graph[block]['function'] + '(' + 'net.parameters()'
            if graph[block]['attributes']:
                optimizer += ',' + fillAttributes(graph, block) + ')\n'
            else:
                optimizer += ')\n'

        # check for dataset
        elif 'dataset' in graph[block]['function']:
            if graph[block]['function'] == 'cifar10_dataset':
                dataset, traintest = datasets.cifar10Dataset()

        # add blocks to the model
        else:
            net += '   ' + graph[block]['function'] + '(' + fillAttributes(graph, block) + '),\n'

    net += ')\n'

    return imports + dataset + net + loss + optimizer + traintest

# fill in the attributes
def fillAttributes(graph, block):
    attributes = ''
    if graph[block]['attributes']:
        for attribute in graph[block]['attributes']:
            if '=' in str(attribute):
                attributes += attribute
            else:
                attributes += str(attribute)
            attributes += ','
        attributes = attributes[:-1]
    return attributes

print(compile(test))
# view generated code
#print(write(test, compile(test)))