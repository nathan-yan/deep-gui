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
    imports = 'import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n'

    # define dataset
    dataset = None

    # define sequential model
    net = 'net = nn.Sequential(\n'

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

# view generated code
print(write(test, compile(test)))