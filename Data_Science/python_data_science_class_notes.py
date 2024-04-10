# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:44:46 2023

@author: Daniel
"""

name = 'alice'
name2 = 'bob'

x = 3
y = 6

xf = 3.6
yf = 6.3

statement = True
statment2 = False


print(type(name))
print(type(x))
print(type(xf))
print(type(statement))

num = '1'
#print('Number type is '+type(num))
num = int(num)
print(type(num))


#%%
######## strings

x = 'This is a string in python!'
#Print string
print(x)
#Print slice of string
print(x[0:5])
# Print reversed string
print(x[::-1])
#Print last character in string 
print(x[-1])
#Print every other
print(x[::2])
print('')



z = x + y
print(str(z))



#%%
#### Data Structures

# Create two sets
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Find their intersection
intersection = set1.intersection(set2)

# Print the result
print("The intersection of set1 and set2 is:", intersection)

# Create two tuples

x = (2,4)
y = (3,6)


# Create a dictionary
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

# Add a new entry
my_dict['pear'] = 4

# Call a value from a key
print("The value of 'banana' is:", my_dict['banana'])


#%%
#### Conditionals

# Define variables x and y
x = 3
y = 6
z = 5


# Check if x is equal to y    
if x == y:
    print("x is equal to y")

# Check if x is not equal to y
if x != y:
    print("x is not equal to y")
    


# Check if x is greater than y
if x > y:
    print("x is greater than y")
# Check if x is less than y
elif x < y:
    print("x is less than y")
# If neither condition is true, then x must be equal to y
else:
    print("x is equal to y")



# Check if x OR y is greater than z    
if x > z or y > z:
    print('x or y is bigger than z')
########################################
# functions

# Declare a simple function
def greet(name):
    print(f"Hello, {name}!")

# Invoke a function
greet('Bob')


### Use return
# Define a function to calculate area and perimeter of a rectangle
def rectangle_calculations(width, height):
    area = width * height
    perimeter = 2 * (width + height)
    return area, perimeter

# Call the function and get the returned values
a, p = rectangle_calculations(5, 10)
print(f"Area: {a}, Perimeter: {p}")



#%%
################################################
# For loops
    
# Loop over a list
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# Loop using range
for i in range(1, 6):
    print(i)



# Enumerate a list
my_list = ['apple', 'banana', 'orange']
for index, item in enumerate(my_list):
    print(index, item)

# Create a list using list comprehension
my_list = [x**2 for x in range(1, 6)]
print(my_list)


# Use a while loop with a counter variable
counter = 0
while counter < 5:
    print(counter)
    counter += 1


# Use a for loop with break, pass, and continue statements
my_list = [1, 2, 3, 4, 5]

# Iterate over the list
for item in my_list:
    # Skip even numbers
    if item % 2 == 0:
        continue
    # Print odd numbers
    print(item)
    # Stop after reaching 3
    if item == 3:
        break
    # Placeholder for future code
    pass

#%%
### Error handle

l1 = [1,2,'tree',5]
l2 = [4,3,7,2]

for entry in zip(l1,l2):
    try:
        print(entry[0]+entry[1])
    except:
        print('error with'+ str(entry))

def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
        return None
    except TypeError:
        print("Error: Please provide valid numeric inputs.")
        return None
    else:
        return result

# Test cases
print(divide_numbers(10, 2))   # Output: 5.0
print(divide_numbers(5, 0))    # Output: Error: Cannot divide by zero. None
print(divide_numbers("a", 2))  # Output: Error: Please provide valid numeric inputs. None

#%%
## Class creation
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Creating objects of class Person
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# Accessing object attributes
print(person1.name)    # Output: Alice
print(person2.age)     # Output: 25

# Calling object method
person1.greet()   # Output: Hello, my name is Alice and I am 30 years old.
person2.greet()   # Output: Hello, my name is Bob and I am 25 years old.



#%%
#Decorator example

def log_greeting(f):
    def wrapped_func(self):
        print(f"Log: {self.name} is about to greet.")
        f(self)
        print(f"Log: {self.name} has finished greeting.")
    return wrapped_func

class Person:
    def __init__(self, name='Default', age=44):
        self.name = name
        self.age = age

    @log_greeting
    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Now when you create a Person object and call greet, it will be logged:
p = Person('Alice', 30)
p.greet()







#%%



#%%


import pandas as pd
import random


#https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

my_df = pd.read_csv('C:\\Users\\Daniel\\Downloads\\archive\\car data.csv')


#%%

# Data types

import pandas as pd
import numpy as np

# Creating a DataFrame with each of the requested data types
df = pd.DataFrame({
    'Object_Column': ['Text', 'More Text', 'Another Text', 'Final Text'],
    'Int_Column': [1, 2, 3, 4],
    'Float_Column': [1.1, 2.2, 3.3, 4.4],
    'Bool_Column': [True, False, True, False]
})

print(df.dtypes)
print('')
print(df.Float_Column.dtype)


#%%
print(my_df.describe())
print(my_df.info())
print(my_df.columns)

print(my_df.Year.describe())
print(my_df['Year'].describe())

print(my_df.Fuel_Type.describe())
#print(my_df['Year'].describe())


#%%

# Replace data

df.Object_Column.replace('Final Text','Really, I am done now', inplace=True)

#%%

# Merge example
df_2 = my_df[['Year','Selling_Price']]
df_3 = my_df[['Kms_Driven','Transmission']]
# Merge the DataFrames on their index
merged_df = df_2.merge(df_3, left_index=True, right_index=True)

# Merge the DataFrames on the 'ID' column
#merged_df = df1.merge(df2, on='ID')

import uuid

# Generate a new column with unique IDs
my_df['UniqueID'] = [uuid.uuid4() for _ in range(len(my_df))]

#%%
#print(my_df['Present_Price'].quantile())

import pandas as pd

# Print correlation
corr = my_df['Year'].corr(my_df['Kms_Driven'])
print(f"Correlation between Year and Kms_Driven: {corr}")




#%%
# Data shape + Slice
print(my_df.dtypes)
# Change type
my_df.astype()

print(my_df.shape)



print(my_df.head())
print(my_df.sample(n=5))


#%%

# missing data

print(my_df.isnull().sum())

my_df.fillna()
# Fill missing values with the mean of each column

#Float,Int
column_means = my_df.select_dtypes(include=['float','int']).mean()
my_df.fillna(column_means, inplace=True)

#Objects
column_modes = my_df.select_dtypes(include=['object']).mode().iloc[0]
my_df.fillna(column_modes,inplace = True)

my_df.dropna()
#drop rows if value is missing from subset column
my_df.dropna(subset=['Present_Price'], inplace=True)
## Example drop data by threshold
# Drop rows with 3 or more missing values
my_df.dropna(thresh=3, inplace=True)

#%%


from sklearn.preprocessing import MinMaxScaler

# Select columns for normalization and scaling
columns_to_normalize = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven']

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Fit and transform the selected columns
my_df[columns_to_normalize] = scaler.fit_transform(my_df[columns_to_normalize])

# Display the updated DataFrame
print(my_df)




#%%
## Filtering data

# filter rows where the Transmission column contains "Manual"
filtered_df_1 = my_df.loc[my_df['Transmission'] == 'Manual']

# filter rows where the Kms Driven column is less than 15000
filtered_df_2 = filtered_df_1.loc[filtered_df_1['Kms_Driven'] < 15000]



#%%


## Data cleaning

## small script to add missing values
# iterate over each column in the DataFrame
for col in my_df.columns:
    
    # iterate over each row in the column
    for i, val in enumerate(my_df[col]):
        
        # randomly set some values to NaN
        if random.random() < 0.05:
            my_df.at[i, col] = pd.np.nan



print(my_df.isnull().sum().sort_values(ascending=False))





#Look at all the unique values with the missing data

for col in my_df.columns:
    print('NANs    ',my_df[col].isnull().sum()) 
    print(my_df[col].value_counts())
    print('')



##

#%%

# Handling Outliers

# Calculate the 95th percentile (quantile) of 'Kms_Driven' to detect outliers
quantile_95 = my_df['Kms_Driven'].quantile(0.95)

# Clip 'Kms_Driven' values to be within the 5th and 95th percentile range
my_df['Kms_Driven'] = my_df['Kms_Driven'].clip(lower=my_df['Kms_Driven'].quantile(0.05), upper=quantile_95)


# Or
# my_df = my_df[my_df['Kms_Driven'] <= 80000]

#%%

## Visualize data


### Heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix using the DataFrame
correlation_matrix = my_df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap',)
plt.show()

#%%
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(my_df['Present_Price'], my_df['Selling_Price'])
ax.set_xlabel('Present_Price')
ax.set_ylabel('Selling_Price')
ax.set_title('Selling Price vs Present Price')
plt.show()

##
#%%
import seaborn as sns

sns.boxplot(x='Fuel_Type', y='Selling_Price', data=my_df)
plt.title('Selling Price by Fuel Type')
plt.show()

sns.countplot(x='Seller_Type', data=my_df)
plt.title('Count of Seller Types')
plt.show()


#%%

##
import seaborn as sns

sns.regplot(x='Kms_Driven', y='Year', data=my_df)
plt.title('Kms Driven vs Year')
plt.show()


#%%
# Distribution of Selling Prices
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of Selling Prices
sns.histplot(data=my_df, x='Selling_Price', kde=True)
plt.title('Distribution of Selling Prices')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

#%%

#Relationship between Selling Price and Present Price

# Scatter plot with regression line
sns.regplot(data=my_df, x='Present_Price', y='Selling_Price')
plt.title('Relationship between Present Price and Selling Price')
plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.show()

#%%

# Boxplot of Selling Prices by Fuel Type
sns.boxplot(data=my_df, x='Fuel_Type', y='Selling_Price')
plt.title('Selling Prices by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()

#%%

# Count plot of cars by Year
sns.countplot(data=my_df, x='Year')
plt.title('Count of Cars by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-labels if they overlap
plt.show()


#%%

# Scatterplot of Kms Driven vs Selling Price
sns.scatterplot(data=my_df, x='Kms_Driven', y='Selling_Price')
plt.title('Kms Driven vs Selling Price')
plt.xlabel('Kms Driven')
plt.ylabel('Selling Price')
plt.show()

#%%

# Pairplot for numeric features
sns.pairplot(data=my_df.select_dtypes(include=['float64', 'int64']))
plt.show()

#%%
# Pairplot with hue
sns.pairplot(hue='Fuel_Type', data= my_df)
plt.show()
#%%

import numpy as np
# Barplot of Average Selling Price by Transmission Type
sns.barplot(data=my_df, x='Transmission', y='Selling_Price', estimator=np.mean)
plt.title('Average Selling Price by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('Average Selling Price')
plt.show()

#%%

# Violin Plot of Selling Prices by Owner Type
sns.violinplot(data=my_df, x='Transmission', y='Selling_Price')
plt.title('Selling Prices by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Selling Price')
plt.show()

#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'my_df' is your DataFrame

# lmplot to show the relationship between Present_Price and Selling_Price
sns.lmplot(data=my_df, x='Kms_Driven', y='Present_Price', hue='Transmission', aspect=1.5)

plt.title('Relationship between Kms_Driven and Present Price by Transmission')
plt.xlabel('Kms_Driven')
plt.ylabel('Present Price (in thousands)')

# Save the figure
#plt.savefig('lmplot.png')

plt.show()


#%%
# Hypothesis Testing
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

fuel_type_petrol = my_df[my_df['Fuel_Type'] == 'Petrol']['Selling_Price']
fuel_type_diesel = my_df[my_df['Fuel_Type'] == 'Diesel']['Selling_Price']

t_stat, p_value = ttest_ind(fuel_type_petrol, fuel_type_diesel)
print("\nHypothesis Testing:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference in Selling Prices based on Fuel Type.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in Selling Prices based on Fuel Type.")
    
# Data Visualization - Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=my_df, x='Fuel_Type', y='Selling_Price')
plt.title("Fuel Type vs. Selling Price")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.show()

#%%
################################################################################
my_df.drop(['Owner'],axis=1, inplace = True)



### Use sklearn to make a random forest model to predict current price

# Import necessary libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Convert categorical variables into dummy variables
my_df = pd.get_dummies(my_df, columns=['Car_Name','Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Prepare the my_df for the model
X = my_df.drop(['Present_Price'], axis=1) # Feature matrix
y = my_df['Present_Price'] # Target variable

# Split the my_df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training my_df
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

#%%

##############################################################################
# Test model on a random value

# Randomly select a row from the original DataFrame
test_row = my_df.sample(1)

# Drop the Present_Price column from the selected row
new_input = test_row.drop('Present_Price', axis=1)

# Use the trained model to make a prediction on the new input
predicted_price = rf.predict(new_input)

# Print the original and predicted selling prices
print('Actual Present Price:', test_row['Present_Price'].values[0])
print('Predicted Present Price:', predicted_price[0])



#%%
### hypertuning


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Create a Random Forest Regressor object
rf = RandomForestRegressor()

# Define the hyperparameters to tune
params = {'n_estimators': [50, 100, 150, 200],
          'max_depth': [5, 10, 15, 20],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['sqrt', 'log2']}

# Perform grid search cross-validation to find the optimal hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", grid_search.best_params_)

# Predict the test data using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)






# Use the trained model to make a prediction on the new input
predicted_price = best_model.predict(new_input)

# Print the original and predicted selling prices
print('Best Model - Actual Present Price:', test_row['Present_Price'].values[0])
print('Best Model - Predicted Present Price:', predicted_price[0])

#%%


############# Evaluation metric to show accuracy of predicting transmission type

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Encode the categorical variables
le = LabelEncoder()
my_df['Fuel_Type'] = le.fit_transform(my_df['Fuel_Type'])
my_df['Seller_Type'] = le.fit_transform(my_df['Seller_Type'])
my_df['Transmission'] = le.fit_transform(my_df['Transmission'])

# Split the dataset into training and testing sets
X = my_df.drop(['Car_Name', 'Transmission'], axis=1)
y = my_df['Transmission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Create a Random Forest Classifier object
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=43)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
from sklearn.model_selection import cross_val_score

# clf is your classifier or regressor
scores = cross_val_score(rf, X, y, cv=5)  # cv is the number of folds
print("Cross-validated scores:", scores)


#%%
# Classification #1

from sklearn.metrics import confusion_matrix

predictions = rf.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n",cm)


#%%

# Classification #2

from sklearn.metrics import classification_report

predictions = rf.predict(X_test)
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)


#%%

# Classification #3

from sklearn.metrics import roc_auc_score

predictions = rf.predict(X_test)
roc = roc_auc_score(y_test, predictions)
print("AUC Score:\n",roc)

#%%

# Classification #3 AUC with viz
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have a fitted classifier 'clf' and a test set (X_test, y_test)
y_scores = rf.predict_proba(X_test)[:, 1]  # get the probability of the positive class
fpr, tpr, threshold = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



#%%

# Regression #1

from sklearn.metrics import mean_squared_error

# Assume reg is your regressor
predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

#%%

# Regression #2

from sklearn.metrics import r2_score

predictions = rf.predict(X_test)
r2 = r2_score(y_test, predictions)
print("R² Score:", r2)

