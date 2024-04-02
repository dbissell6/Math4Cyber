# Machine Learning

It really doesnt need an intro, you have heard about it everyday. AI is here to replace you. But how exactly does it do it?

## Types of machine learning

When we teach a machine we typically do so in 3 ways. 

1) Supervised Learning

2) Unsupervised Learning

3) Reinforcement Learning


## The flow

Machine learnings first couple steps are the same as using most functions. There is an input, the function functions, there is an output. In machine leanring, during the training phase, the machine output
is compared to the correct answer(For instance, if the machine was being trained to guess a picture of a cat or dog, after it said cat, it would be compared to the answer for the photo. This is a problem with kind of learning,
training takes many examples and each examples answer has to be labled by a person.

```Inputs Weights -> Summation Activation -> Propagation -> Output -> Back Propogation and Weight Adjustment```

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/8c20555e-294f-484d-806c-175a8f7273c6)


To play around with. `https://playground.tensorflow.org/`



## Inputs Weights

## Summation Activation

2 steps here. Get the sum from the edges coming into the neruon(node). Determine if they pass a threshold. 
This activation aspect is crucial as it allows non linearity into the network.

Lets focus on the inputs to a single node.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/961c6c93-5787-4d10-814d-f9308aaed9c9)

Here is the activation part. 

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/16bb0c19-4385-45a9-a6c3-0ccbfad12236)


Some actual activation functions that are used.

ReLU (Rectified Linear Unit): Popular for its simplicity and efficiency. It outputs the input directly if it is positive; otherwise, it outputs zero. It's widely used in hidden layers because it helps with faster training and reducing the likelihood of vanishing gradients.

Tanh (Hyperbolic Tangent): Similar to sigmoid but outputs values between -1 and 1. It's useful for tasks where negative outputs are meaningful.

Sigmoid: Outputs a value between 0 and 1, making it ideal for binary classification tasks, like determining whether an email is spam or not.

Softmax: Used in the output layer for multi-class classification tasks. It converts logits to probabilities by taking the exponential of each output and then normalizing these values by dividing by the sum of all the exponentials.

## Propagation

## Output

The output can be thought of as the algorithms answer, or response. 

## Backpropogation and Weight Adjustment

Imagine backpropagation as the neural network’s moment of self-reflection, where it looks back at its predictions to understand its mistakes. It's the mechanism by which neural networks learn from the error in their predictions. This process involves calculating the gradient (or change) of the loss function with respect to each weight in the network by tracing back from the output layer to the input layer.

### Optimizers

Optimizers are the navigators of your neural network, steering the learning process by adjusting weights to minimize the loss function. They determine how quickly or slowly a network learns.

SGD (Stochastic Gradient Descent): A classic, simple optimizer that updates parameters in the opposite direction of the gradient. It's robust but can be slow and less efficient on complex landscapes.

Adam (Adaptive Moment Estimation): Combines the best properties of two other extensions of SGD, AdaGrad and RMSProp, to handle sparse gradients on noisy problems. It's known for being efficient and effective across a wide range of tasks.

RMSprop (Root Mean Square Propagation): Modifies SGD by dividing the gradient by a running average of its recent magnitude, helping to resolve the vanishing or exploding gradient problems.

### Loss Functions

Loss functions are the compasses of neural networks, they guide the optimizer by indicating the direction to take to improve model performance.

Mean Squared Error (MSE): Commonly used for regression tasks. It measures the average squared difference between the estimated values and the actual value.

Cross-Entropy Loss: Preferred for classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.

Binary Cross-Entropy: A special case of cross-entropy loss for binary classification tasks.
