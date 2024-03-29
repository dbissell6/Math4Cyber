# Multivariable Calculus


## Partial Derivatives

**Partial Derivatives**: These are derivatives of functions with more than one variable, taken with respect to one variable at a time.
In the context of gradient descent, partial derivatives tell you how the function changes as each individual input variable changes, holding all other variables constant.

## Total Differentials

**Total Differentials**: This involves a combination of all the partial derivatives to express the total change in a function for a given change in all of its variables.

## Gradient

**Gradient**: It's the vector of all the partial derivatives of a function.
The gradient points in the direction of the steepest ascent. For gradient descent, we’re interested in going in the opposite direction to find the minimum of a function.


**Step Size**: The learning rate in gradient descent determines how far to move in the direction opposite to the gradient.
This involves a bit of the total differentials concept (found in 2.2) because you need to consider the change in the function with respect to changes in all of its variables.
