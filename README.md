# deep-gui
A graphical deep learning library

## Setup
To install all Gradient dependencies, cd into the Gradient directory, then do:

```
cd backend
pip install -r requirements.txt
```

It might take a while to install everything!

## Intro to Neural Networks
Neural networks have recently become an extremely popular machine learning technique, and are used to solve problems in image classification, natural language processing and robotics, among many others. This section gives a brief overview of how neural networks work; enough to understand how Gradient works at least.

Neural networks take in an input, which is usually a matrix or a _tensor_ (which is just a matrix but with an extra dimension, think of a matrix as a square and a tensor as a cube or even hypercube!), and spit out an output. For example, in the case of image classification, the input of the neural network might be an image of width 100 pixels x 100 pixels. In this case, the input to the network is a 3 x 100 x 100 element tensor (the 3 comes from the R, G and B color channels). Each number in the tensor is the intensity of the R, G or B value of a single pixel. The network output might be a single number, which is the probability that the given image is a dog.

Neural networks have inside them many parameters, which are numbers that tell the network what to do with its given input to produce an output. These parameters start out randomly initialized, which means the network probably will do quite poorly in the beginning. In the context of Gradient, we typically use _supervised learning_ to train networks. 

### Supervised Learning
When we give the network an input as well as the _expected output_ (the correct, or groundtruth answer) of the network, we can see how our network’s output compares to its expected output. From this comparison, we can calculate how to change our parameters using a clever algorithm called backpropagation and gradient descent. If we give the network many of these examples, our parameters will slowly get better and better. 

### Loss Functions
So how does our network compare its output to its expected output? This is a special function called a _loss function_. A common loss function is simply the squared distance between the expected output and the network output. Another loss function, which we’ll use in our example, is categorical cross entropy. This has a fancy definition, but what it is in simple terms is the natural log of the probability our network picks the correct answer, if it samples its answers from its output. 

## How to use Gradient
Typically, neural networks are created using libraries like PyTorch or Tensorflow, and require you to actually code the network. Gradient attempts to turn this coding-only approach into a hybrid of coding and graphics. You create the network using a graphical interface, and insert that network into your code wherever you want. This maintains the flexibility of coding while allowing you to develop a network extremely quickly using drag and drop blocks.

WARNING: Since Gradient is a work in progress, there is no ability to SAVE your network. That is, if you ever refresh your browser, your network progress will be erased, and you will have to start over again. This is a feature we’re hoping to add soon! 

To start creating a network, open the application by doing 

```
cd backend
python server.py
```

This will run the server as well as open up the web application in your default browser.

On the left there will be a series of blocks to choose from. This is currently a pretty limited list. On the right there is the parameter editor, where you can change the behavior of certain blocks. In the center is the actual graph editor, where you can drag and drop blocks, and connect blocks together.

We’ll start by creating a simple feedforward neural network and training it on the MNIST dataset (MNIST is a dataset of handwritten digits from 0 - 9, our network’s goal is to correctly classify these digits). 

First, drag a “flatten” layer into the editor. The flatten layer transforms your 1 x 28 x 28 tensor into a 784 vector. This is because our next layer requires a vector input, not a tensor input. Your editor should look like this:


Next, drag a “dense” layer into the editor. Drag the “output” bubble from the “flatten” layer to the “input” bubble of your “dense” layer. This will connect the two. You’ll notice that it has two inputs under the block, named “weights” and “input”. The “weights” input is shaded black because it has parameters you can edit. You do NOT need to connect anything to an input that is shaded black. Click on the “weights” input to edit its parameters on the right of the screen. Go ahead and change the in_features value to 28\*28. The reason why our value is 28\*28 is because the digit images are 28 by 28 pixels in size. Next, change the out_features value to 10. Out_features defines how many output values our dense layer produces. The 10 output values correspond to how likely it thinks the input image is each digit from 0 - 9. As an example, if the 5th output value is especially high, the network is more confident that the image is a 5. Once you are done editing your parameters, press “enter” to save them all. Your editor should look like this:



The final step is to add a softmax layer. A softmax layer normalizes the output of your dense layer so that they add up to 1. The reason for this is that your network is actually outputting probabilities. Each output is the probability that the image is a certain digit. In order for the outputs to be valid probabilities, they must add up to 1. As an example, if our network outputs the following probabilities:

```
0    1    2    3    4    5    6    7    8    9

0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0
```

The network thinks there is a 20% chance the image is a 0, a 20% chance the image is a 1, and a 10% chance that the image is a 2, 3, 4, 5, 6, or 7. 

To add the softmax, just drag a softmax layer and connect the output of the dense layer to the input of the softmax, by dragging from the “output” bubble of the dense layer to the “input” bubble of the softmax. Next, click on the “dimension” input of the softmax, and change “dim” to 1. Your editor should now look like this:



Congratulations! It may not look like it, but you’ve made your first (albeit very simple) neural network in Gradient! To actually compile your network, click on the “compile ->” button on the top right. To train and test your network, go to the example directory we’ve made by opening another terminal, navigating to the Gradient directory and entering the commands:

`cd example_workspace`
`python compiled.py`

This will run your compiled file. If everything works nothing should happen for a while. After a minute or two the network will finish training and you should get output like this:



It looks like our simple network has accuracy of 88%. It is most accurate at detecting 0s (97%), and least accurate at detecting 5s (61%). The confusion matrix below tells you what digits are most commonly confused for others. For example, the row beginning with the digit ‘5’ and the column beginning with the digit ‘3’ is 158. This means that the digit 5 was confused for the digit 3 158 times during testing. This makes sense, since 5s actually look pretty similar to 3s. Take a look at the confusion matrix of your network and see if you can find any interesting insights!
