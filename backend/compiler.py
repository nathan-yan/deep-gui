# test json
test = {
   "Conv2d_1":{
      "outputs":[
         "ReLU_1"
      ],
      "parameters":[
         1,
         20,
         5
      ],
      "function":"Conv2d"
   },
   "ReLU_1":{
      "outputs":[
         "Conv2d_2"
      ],
      "parameters":[
         
      ],
      "function":"ReLU"
   },
   "Conv2d_2":{
      "outputs":[
         "ReLU_2"
      ],
      "parameters":[
         20,
         64,
         5
      ],
      "function":"Conv2d"
   },
   "ReLU_2":{
      "outputs":[
         
      ],
      "parameters":[
         
      ],
      "function":"ReLU"
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
    code = "import torch\nimport torch.nn as nn\n\nmodel = nn.Sequential(\n"

    while order:
        block = order.pop()
        code += getLayer(graph, block)
    
    return code + ")"

# get layer code
def getLayer(graph, block):
    layer = 'nn.' + graph[block]['function'] + '(' + str(graph[block]['parameters'])[1:-1] + '),\n'
    return layer

# view generated code
print(write(test, compile(test)))