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

<img width="252" alt="Screen Shot 2024-05-09 at 9 49 33 AM" src="https://github.com/dbissell6/Math4Cyber/assets/50979196/da4e6aa5-2b30-45bc-b9ab-70f77d9b442d">

## Random Forests

A Random Forest is an ensemble learning technique that builds on the simplicity of decision trees and enhances their performance and accuracy. It consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest predicts the outcome (class or value), and the class with the most votes becomes the model’s prediction. Random Forests also help to reduce overfitting.

## Gradient Boosting

Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, but it generalizes them by allowing optimization of an arbitrary differentiable loss function.

`Sequential Building`: Unlike Random Forests, which build trees in parallel, gradient boosting builds one tree at a time. Each new tree helps to correct errors made by previously trained trees.

`Loss Function Optimization`: Each tree is trained using the gradient (hence the name) of the loss function, which measures how well the model predicts the training data. By focusing on minimizing the loss, gradient boosting systematically improves the model's performance.

`Flexibility`: It can be used with different loss functions and hence can be adapted for various prediction problems, including both regression and classification tasks.

## XGBoost (Extreme Gradient Boosting)

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It has become one of the dominant machine learning algorithms in competitive machine learning because of its efficiency at handling large datasets and its effectiveness across a wide range of predictive tasks.

`Efficiency at Scale`: XGBoost is specifically designed to be efficient with large datasets, utilizing both hardware optimization and software improvements.

`Regularization`: It includes L1 and L2 regularization, which improves model generalization capabilities and helps to prevent overfitting.

`Handling Missing Values`: XGBoost has an in-built routine to handle missing data. When the model encounters a missing value at a node, it uses a default direction to decide the split based on what was learned during the training.

`Tree Pruning`: Unlike traditional gradient boosting, which stops growing trees when they reach a maximum depth, XGBoost grows the tree up to a maximum depth and then prunes it back to minimize the loss, potentially removing unnecessary splits.

`Built-in Cross-Validation`: XGBoost allows users to run a cross-validation at each iteration of the boosting process, making it easy to obtain accurate models that do not overfit the training data.

# Bayesian models 

Bayesian models are based on Bayes' Theorem, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event. These models are used for a wide range of tasks including classification, regression, and prediction. Bayesian models are particularly known for their ability to provide probabilistic predictions, which means they can tell you how confident they are about their predictions. They are incredibly useful in scenarios where the data is incomplete or uncertain, as they can incorporate prior knowledge into the model. They can adaptively update themselves with new evidence, making them very flexible.


# Neural Networks (NN)

Neural Networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or raw input processing. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text, or time series, must be translated.

## Components of a Neuron

Neural Networks are non-linear models. This non-linearity arises primarily from the activation functions used in the neurons of the network.

### Inputs

These are the data points fed into the neuron. Each input will have an associated weight, reflecting the importance or strength of the input in determining the output.

### Weights

Weights are parameters that are learned during the training of the network. They scale the input data, amplifying or dampening the effect of inputs based on their relevance to the task.

### Bias 

Bias allows the activation function to be shifted to the left or right, which can be critical for learning complex patterns.

### Summation Function

This function sums up all the inputs multiplied by their respective weights and adds the bias. This sum is then passed to the activation function. It's often just a simple weighted sum.

### Activation Function

This function takes the output of the summation function and applies a non-linear transformation, deciding how much and whether the signal should proceed further through the network.

### Output

The final output of the neuron after the activation function has been applied. This output can then be used as an input for subsequent layers in a neural network or as the final output for the last layer depending on the architecture.


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

Backpropgation consists of two passes, the forward pass and backward pass.

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


# Additional concerns

We typically say the model overfits the data if the model predicts accurately on the training dataset but doesn’t generalize well to other test examples, that is, if the training error is small but the test error is large. We say the model underfits the data if the training error is relatively large.

Overftting is typically a result of using too complex models, and we need to choose a proper model complexity to achieve the optimal bias-variance tradeoff. Regularization is 
used to control the complexity and minimize overfitting.







