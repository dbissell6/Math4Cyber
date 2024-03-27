# Linear Algebra

Linear algebra is the branch of mathematics that deals with vectors, matrices, vector spaces (also known as linear spaces), linear transformations, and systems of linear equations.


## Applications

Cryptography: This is probably the most direct application of linear algebra in cybersecurity. Many cryptographic algorithms, including RSA, Elliptic Curve Cryptography (ECC), and more, rely on the complex mathematical properties that linear algebra provides. For example, the difficulty of solving certain linear algebra problems (like finding the eigenvalues of a matrix) underpins the security of some encryption methods.

Data Analysis and Machine Learning: Linear algebra is crucial for understanding and designing algorithms used in machine learning and data analysis. These algorithms can be used for threat detection, analyzing network traffic, identifying anomalies, and making predictive analyses regarding potential security breaches.

Network Theory: The analysis of networks, whether they are computer networks, social networks, or otherwise, often uses matrices and other linear algebra concepts to represent and analyze the relationships between nodes in a network. This can be crucial for understanding the spread of malware or the structure of botnets.

## Definitions

A vector is a mathematical object that has both magnitude and direction.

A matrix is a rectangular array of numbers, symbols, or expressions, arranged in rows and columns. Matrices are used to perform linear transformations, solve systems of linear equations, and represent data. Each number in a matrix is called an element. The size of a matrix is defined by its number of rows and columns, often referred to as m×nm×n, where mm is the number of rows, and nn is the number of columns.

A linear transformation is a mapping between two vector spaces that preserves the operations of vector addition and scalar multiplication. Essentially, if you apply a linear transformation to a vector, its direction and/or magnitude are changed in a way that is consistent and predictable across the space. Linear transformations can be represented by matrices, making the connection between these concepts very direct.


Start with vectors

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/2e167811-ccc0-47e1-8cad-5ceb42e093b2)

Above are examples of 2 vectors. They are 2 dimesional, plotted on a 3d space(z = 0). 

One important fact here is matrix multiplication, we can multiply a metrix by a vector to get a new vector. what is important here is if we get the inverse of the matrix we can use it to decode the message, or get the original position.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/07ebface-effd-4711-8c8d-e90ee46e1bf3)

Above first part is the matrix, second is the vector, 3rd is the result of multiplication. 

Step towards thinking how these concepts can be converted and applied to cryptography. 

Here are the same two dimensional vectors. On the axis we have 0,26. I have done this so each number can correspond with a letter. a=1,b=2,...z=26. In this 2 dimensional space we could have a vector that would correspond with every 2 letter word. 

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/95f90afd-6be8-4a02-ac62-4658f2004f38)


This is proper vector notation. In our case the would be the vectors BE and ME

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/c952f932-6acc-4bb6-988e-f22e707d16a3)


Thinking back to the vector x matrix.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/28f3af9d-22ed-478e-a0a7-6b3d18ec5e19)


Going up to 3 dimensions. This is the last stage that is easy to visualize. importnat part to know, even though we cant visualize more than this, the computations are the same, it makes no difference to the computer.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/cfca6038-d9f0-4a1c-95d0-213e521d1367)

