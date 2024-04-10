# Machine Learning

It really doesnt need an intro, you have heard about it everyday. AI is here to replace you. But how exactly does it do it?

# Types of machine learning

When we teach a machine we typically do so in 3 ways. 

1) Supervised Learning 

2) Unsupervised Learning

3) Reinforcement Learning


The big distinction between supervised learning and unsupervised learning is the target. I supervised learning someone must go through all of the training data and label the or thing the model is trying to predict. For instance to train a model to determine if an image was that of a cat or dog, we would have to use a supervised learning and someone would have to go through every image and label if it was a cat or dog. 

Unsupervised learning is different in that it will create groups for you. 

Finally reinforement learning is good for scenerios like games. This model works by taking action and if the reward is increased, the model 'remembers that' (state,action). The easiest thing to imagine is a videogame score. 

# Types of models

# Decision Trees

Decision Trees are a type of supervised learning algorithm that can be used for both classification and regression tasks. They work by breaking down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node has two or more branches, each representing values for the attribute tested. Leaf nodes represent a classification or decision. The topmost decision node in a tree corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data and are simple to understand and interpret.


# Bayesian models 

Bayesian models are based on Bayes' Theorem, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event. These models are used for a wide range of tasks including classification, regression, and prediction. Bayesian models are particularly known for their ability to provide probabilistic predictions, which means they can tell you how confident they are about their predictions. They are incredibly useful in scenarios where the data is incomplete or uncertain, as they can incorporate prior knowledge into the model. They can adaptively update themselves with new evidence, making them very flexible.


# Neural Networks (NN)

Neural Networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or raw input processing. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text, or time series, must be translated.

## The Neural Network Flow

```Inputs Weights -> Summation Activation -> Propagation -> Output -> Backpropogation and Weight Adjustment```

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

`ReLU (Rectified Linear Unit)`: Popular for its simplicity and efficiency. It outputs the input directly if it is positive; otherwise, it outputs zero. It's widely used in hidden layers because it helps with faster training and reducing the likelihood of vanishing gradients.

`Tanh (Hyperbolic Tangent)`: Similar to sigmoid but outputs values between -1 and 1. It's useful for tasks where negative outputs are meaningful.

`Sigmoid`: Outputs a value between 0 and 1, making it ideal for binary classification tasks, like determining whether an email is spam or not.

`Softmax`: Used in the output layer for multi-class classification tasks. It converts logits to probabilities by taking the exponential of each output and then normalizing these values by dividing by the sum of all the exponentials.

## Propagation

## Output

The output can be thought of as the algorithms answer, or response. 

### Regression

### Classification

## Backpropogation and Weight Adjustment

Imagine backpropagation as the neural network’s moment of self-reflection, where it looks back at its predictions to understand its mistakes. It's the mechanism by which neural networks learn from the error in their predictions. This process involves calculating the gradient (or change) of the loss function with respect to each weight in the network by tracing back from the output layer to the input layer.

### Optimizers

Optimizers are the navigators of your neural network, steering the learning process by adjusting weights to minimize the loss function. They determine how quickly or slowly a network learns.

`SGD (Stochastic Gradient Descent)`: A classic, simple optimizer that updates parameters in the opposite direction of the gradient. It's robust but can be slow and less efficient on complex landscapes.

`RMSprop (Root Mean Square Propagation)`: Modifies SGD by dividing the gradient by a running average of its recent magnitude, helping to resolve the vanishing or exploding gradient problems.

`Adam (Adaptive Moment Estimation)`: Combines the best properties of two other extensions of SGD, AdaGrad and RMSProp, to handle sparse gradients on noisy problems. It's known for being efficient and effective across a wide range of tasks.

### Loss Functions

`Loss functions` are the compasses of neural networks, they guide the optimizer by indicating the direction to take to improve model performance.

`Mean Squared Error (MSE)`: Commonly used for regression tasks. It measures the average squared difference between the estimated values and the actual value.

`Cross-Entropy Loss`: Preferred for classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.

`Binary Cross-Entropy`: A special case of cross-entropy loss for binary classification tasks.
