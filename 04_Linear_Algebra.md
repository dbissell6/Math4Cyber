# Linear Algebra

Linear algebra is the branch of mathematics that deals with vectors, matrices, vector spaces (also known as linear spaces), linear transformations, and systems of linear equations.


## Applications

Cryptography: This is probably the most direct application of linear algebra in cybersecurity. Many cryptographic algorithms, including RSA, Elliptic Curve Cryptography (ECC), and more, rely on the complex mathematical properties that linear algebra provides. For example, the difficulty of solving certain linear algebra problems (like finding the eigenvalues of a matrix) underpins the security of some encryption methods.

Data Analysis and Machine Learning: Linear algebra is crucial for understanding and designing algorithms used in machine learning and data analysis. These algorithms can be used for threat detection, analyzing network traffic, identifying anomalies, and making predictive analyses regarding potential security breaches.

Network Theory: The analysis of networks, whether they are computer networks, social networks, or otherwise, often uses matrices and other linear algebra concepts to represent and analyze the relationships between nodes in a network. This can be crucial for understanding the spread of malware or the structure of botnets.

## Definitions

A **vector** is a mathematical object that has both magnitude and direction.

A **matrix** is a rectangular array of numbers, symbols, or expressions, arranged in rows and columns. Matrices are used to perform linear transformations, solve systems of linear equations, and represent data. Each number in a matrix is called an element. The size of a matrix is defined by its number of rows and columns, often referred to as m×n m×n, where m is the number of rows, and n is the number of columns.

A **linear transformation** is a mapping between two vector spaces that preserves the operations of vector addition and scalar multiplication. Essentially, if you apply a linear transformation to a vector, its direction and/or magnitude are changed in a way that is consistent and predictable across the space. Linear transformations can be represented by matrices, making the connection between these concepts very direct.

3Blue1Brown "Essence of Linear Algebra" ```https://youtu.be/fNk_zzaMoSs?si=ZXSooq0jqIq1XweX```

Start with vectors

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/2e167811-ccc0-47e1-8cad-5ceb42e093b2)

Above are examples of 2 vectors. They are 2 dimesional, plotted on a 3d space(z = 0). 

## Multiplication

One important fact we need to know before moving forward is matrix multiplication, we can multiply a matrix by a vector to get a new vector. 

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/07ebface-effd-4711-8c8d-e90ee46e1bf3)

Above first part is the matrix, second is the vector, 3rd is the result of multiplication. 

Very abstract, start plugging in some numbers and introduce key fact, What is important here is if we get the inverse of the matrix and multiply it but that new vector, we can get the original vector back.

Top row is showing basic muliplication of a vector and matrix resulting in that new vector. Row below is just reversing the process. Another key fact, if we know the og matrix we can calculate its inverse.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/8623f8e5-8230-40e9-a81f-67244f665d94)

## Towards encryption

Step towards thinking how these concepts can be converted and applied to cryptography. 

Before we apply linear algebra concepts to encryption lets think about a very simple type encryption. Imagine you and a friend want to sent notes to each other but you dont want anyone that intercepts it to be able to read it. You come up with an idea each letter will equal a number a=1,b=2,c=3, so

8,9 = HI
2,25,5 = BYE

Imagine this works for a bit but someone takes the note and is able to decipher the code. So you and your friend decide to 'shift' the numbers by multipling them by a key. you choose the key = 3. You multiply the numbers by the key, what does the person need to do to decrypt it? divide by 3, or multiply by the inverse 1/3.

3*8,3*9 = 24,27 
3*2,3*25,3*5 = 6,75,25

Here are the same two dimensional vectors with a bunch more added. The dont have the lines and arrows, but they can be interpreted the same.

On the axis we have 0,26. I have done this so each number can correspond with a letter. a=1,b=2,...z=26. In this 2 dimensional space we could have a vector that would correspond with every 2 letter word. 

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/95f90afd-6be8-4a02-ac62-4658f2004f38)


This is proper vector notation. In our case the would be the vectors BE and ME

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/c952f932-6acc-4bb6-988e-f22e707d16a3)


Thinking back to the vector x matrix.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/28f3af9d-22ed-478e-a0a7-6b3d18ec5e19)

The nice thing about the mult is that the process is reversable

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/e69b6822-ce67-4bb9-8759-03ce4cce2d9f)


Going up to 3 dimensions. This is the last stage that is easy to visualize. importnat part to know, even though we cant visualize more than this, the computations are the same, it makes no difference to the computer.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/cfca6038-d9f0-4a1c-95d0-213e521d1367)


### Python implementation

Alright we know understand the basic concepts of linear algebra and can see a very basic applciation, lets expand.

In the orginal presentation we used 0-26, that helps make intuitive sense for the mapping. Here we use what the computer knows, ASCII.

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/4adc5398-e80f-4827-adce-88e232b31370)

As expected we can reverse the process

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/68ae4fc0-e62c-4e4b-a1db-184612187734)


It will work for longer strings too

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/649eceb7-25c1-4c0e-887b-dfc7edb58384)

What have we done? We created a encryption mechanism using the principles of linear algebra and matricies. 
We can pass(matrix multiplication) a message(vector) through a filter(matrix) and get an encoded messege. Becasue we can multiply with the same amount of colums we had to split the vector into checks that matched the number of colums in the matrix. 

In order to decode the encoded message we need to get the inverse of the original matrix. We can multiply this with the encoded message to get the original.(Identity Matrix) 

What cant we do? We instantiated our matrix with seemingly random values, could any values work? no! all 0 in a column or row  will kill us.  
In order for the matrix to work for encryption the determinant must be non-zero. This should be simple to understand. Anything multiplied by 0 ends up equaling 0. Therefore we cant reverse this. 

**Determinant**

The **determinant** of a square matrix is a scalar value that reflects the matrix's invertibility, scales areas or volumes under linear transformations, and is crucial in solving linear equations. A non-zero determinant indicates an invertible matrix, while a determinant of zero means the matrix is singular (non-invertible).

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/786c8b8f-cbb4-4fe6-81ae-5435fb01cb87)

Above will give us the same issue. The determinate of this matrix is 0 and is said to be singuar. This transformation loses the original distinctiveness of x and y, collapsing any input into a less complex space where x=y. Again a loss of information. 

Anything else?

Matrices with Repeated Rows or Columns

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/ce9cca56-8309-4600-b9af-5f99903bf55c)


A matrix with repeated rows or columns (e.g., above) is singular and thus not invertible. These matrices fail to span the entire space they're supposed to, limiting their ability to uniquely transform vectors for encryption purposes.

Anything else?
one or more zeros on its main diagonal


The matrix acts like the key in modern day encryption. 


<details>

<summary>Python encryption code</summary>

```
import numpy as np

def string_to_ascii_vector(s):
    return [ord(c) for c in s]

def ascii_vector_to_string(v):
    return ''.join(chr(int(round(i))) for i in v)

def encode_chunks(message, matrix):
    # Ensure message length is even for 2x2 matrix
    if len(message) % 2 != 0:
        message += " "  # Padding if necessary
    
    encoded_message = []
    decoded_message = []
    
    # Process in chunks matching the matrix size (2 for 2x2 matrix)
    for i in range(0, len(message), 2):
        chunk = message[i:i+2]
        vector = np.array(string_to_ascii_vector(chunk))
        encoded_vector = np.dot(matrix, vector)
        encoded_message.extend(encoded_vector)
        
    print(f"Original Message in ASCII: {vector}")
    return encoded_message

# Define matrices
matrix = [[1, 2], [3, 4]]
transformation_matrix = np.array(matrix)
inverse_matrix = np.linalg.inv(transformation_matrix)

# Original message
original_message = "Vivis{Ghost_In_The_Box}"

# Encode and decode the message
encoded_message = encode_chunks(original_message, transformation_matrix)

# Convert vectors back to string
encoded_string = ascii_vector_to_string(encoded_message)

print("")
print(f"Original Message: {original_message}")
print("")
print(f"Matrix: {matrix}")
print("")
print(f"Encoded Vector: {encoded_message}")


#%%

import numpy as np

def ascii_vector_to_string(v):
    # Ensure rounding to the nearest int since ASCII values are integers
    return ''.join(chr(int(round(i))) for i in v)


def get_chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


# Define the inverse of the transformation matrix used for encoding
# (This would be provided or you'd need to calculate it)
#matrix = [[1, 2], [3, 4]]
inverse_matrix = np.linalg.inv(np.array(matrix))

# The encoded vector (split into chunks that match the matrix size)
# Let's use the vector from your screenshot as an example
encoded_vector = encoded_message

chunk_size = 2  # Size of the matrix used for encoding

# Get the chunks as a list to use for decoding
encoded_vector_chunks = list(get_chunks(encoded_vector, chunk_size))

# Decoding each chunk
decoded_message = ""
for chunk in encoded_vector_chunks:
    decoded_vector = np.dot(inverse_matrix, chunk)
    decoded_message += ascii_vector_to_string(decoded_vector)

print(f"Encoded Message: {encoded_vector}")
print("")
print(f"Matrix: {matrix}")
print("")
print(f"Inverse Matrix: {inverse_matrix}")
print("")
print(f"Decoded Message: {decoded_message}")
```

</details>


