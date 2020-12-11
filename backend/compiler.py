import datasets
import addTorch

# order blocks
def compile(graph, inputs):
   orderedBlocks = []
   compiledBlocks = {}
   for block in graph:
      compiledBlocks[block] = False
   for block in graph:
      if not compiledBlocks[block]:
         topologicalSort(graph, block, inputs, orderedBlocks, compiledBlocks)
   return orderedBlocks, inputs

# recursively stack blocks in order
def topologicalSort(graph, key, inputs, stack, visited):
   visited[key] = True
   for value in graph[key]['inputs']:
      if graph[key]['inputs'][value] == None:
         if key + "." + value not in inputs:
            print("invalid")
            break;

         #inputs.append(key + "_" + value)
         graph[key]['inputs'][value] = key + "." + value

      elif not visited[graph[key]['inputs'][value].split('.')[0]]:

         print(value, graph[key])
         topologicalSort(graph, graph[key]['inputs'][value].split('.')[0], inputs, stack, visited)
   stack.append(key)

# write
def write(file, graph):
   with open(file, 'r') as f:
      contents = f.read()
      lines = contents.split('\n')

   for line in lines:
      
      # TODO: this is pretty poor parsing, should use something else
      if '@network' in line:
         start = line.index('(')
         end = line.index(')')

         content = line[start : end]
         input_start = line.index("[")
         input_end = line.index("]") + 1

         input_content = line[input_start + 1 : input_end - 1]

         output_start = line[input_end:].index("[")
         output_end = line[input_end:].index("]") + 1

         output_content = line[input_end + output_start + 1 : input_end + output_end - 1]

         inputs = input_content.split(',')
         outputs = output_content.split(',')

         inputs = [i.replace(' ', '')[1:-1] for i in inputs]
         outputs = [o.replace(' ', '')[1:-1] for o in outputs]

         order, inps = compile(graph, inputs)
      
         # define network
         net = 'class Net(nn.Module):\n   def __init__(self):\n      super(Net, self).__init__()\n'

         # define forward function
         forward = '\n   def forward(self, '
         for i in inps:
            forward += i.replace(".", "_") + ", "
            
         
         forward = forward[:-2] + "):\n"

         # define initialize model
         initialize = '\nnet = Net()\n'

      
         # generate script from stack
         while order:
            block = order.pop(0)
         
            if 'add' == graph[block]['type']:
               addNet, addForward = addTorch.addSum(block, graph[block]['inputs'], graph[block]['attributes'])

               net += addNet
               forward += addForward

            # add blocks to the model
            elif 'conv2d' == graph[block]['type']:
               addNet, addForward = addTorch.addConv2d(block, graph[block]['inputs'], graph[block]['attributes'])
               net += addNet
               forward += addForward
            
            elif 'maxpool' == graph[block]['type']:
               addNet, addForward = addTorch.addMaxPool2d(block, graph[block]['inputs'], graph[block]['attributes'])
               net += addNet
               forward += addForward
               
            elif 'flatten' == graph[block]['type']:
               addNet, addForward = addTorch.addFlatten(block, graph[block]['inputs'], graph[block]['attributes'])
               net += addNet
               forward += addForward

            elif 'dense' == graph[block]['type']:
               addNet, addForward = addTorch.addLinear(block, graph[block]['inputs'], graph[block]['attributes'])
               net += addNet
               forward += addForward
            
            elif 'relu' == graph[block]['type']:
               forward += addTorch.addReLU(block, graph[block]['inputs'], graph[block]['attributes'])
            
            elif 'sigmoid' == graph[block]['type']:
               forward += addTorch.addSigmoid(block, graph[block]['inputs'], graph[block]['attributes'])

            elif 'tanh' == graph[block]['type']:
               forward += addTorch.addReLU(block, graph[block]['inputs'], graph[block]['attributes'])
            
            elif 'softmax' == graph[block]['type']:
               forward += addTorch.addSoftmax(block, graph[block]['inputs'], graph[block]['attributes'])

         forward += "\n      return "
         for o in outputs:
            forward += o.replace(".", "_") + ", "

         forward = forward[:-2]

         contents = contents.replace(line, net + forward + initialize)

   return contents
   #return net + forward + initialize

if __name__ == '__main__':
   # test
   test = {"conv_1":{"inputs":{"input":None},"attributes":["in_channels=3","out_channels=6","kernel_size=[3, 3]","stride=[1, 1]","padding=[1, 1]","dilation=[1, 1]","groups=1","bias=True","padding_mode=zeros"],"type":"conv2d"},"conv_2":{"inputs":{"input":"relu_1.output"},"attributes":["in_channels=6","out_channels=12","kernel_size=[3, 3]","stride=[1, 1]","padding=[1, 1]","dilation=[1, 1]","groups=1","bias=True","padding_mode=zeros"],"type":"conv2d"},"relu_1":{"inputs":{"input":"conv_1.output"},"attributes":[],"type":"relu"},"add_1":{"inputs":{"input1":"conv_2.output","input2":"conv_1.output"},"attributes":[],"type":"add"},"tanh_1":{"inputs":{"input":"add_1.output"},"attributes":[],"type":"tanh"}}

   #print(addTorch.addConv2d('conv_2', test['conv_2']['inputs'], test['conv_2']['attributes']))
   #print(addTorch.addSum('add_1', test['add_1']['inputs'], test['add_1']['attributes']))

   # view generated code

   print(write('blank.py', test))
