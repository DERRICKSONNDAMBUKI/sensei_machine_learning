# dataset from UCI: https://archive.ics.uci.edu/ml/datasets/student+performance
# We download the ZIP-file from the Data Folder and extract the file student-
# mat.csv from there into the folder in which we code our script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('./student/student-mat.csv', sep=';')
# print(data)
data = data[['age', 'sex', 'studytime', 'absences', 'G1', 'G2', 'G3']]
# The columns G1, G2 and G3 are the three grades that the students get.
# print(data)
# Our goal is to predict the third and final grade by looking at the other values like
# first grade, age, sex and so on.

data['sex'] = data['sex'].map({
    'F': 0,
    'M': 1
})
# print(data)
# we define the column of the desired label as a variable to make it
# easier to work with.
prediction = 'G3'
# The sklearn models do not accept Pandas data frames, but only NumPy arrays. That's why
# we turn our features into an x-array and our label into a y-array.

X = np.array(data.drop([prediction],1))
Y = np.array(data[prediction])

# The method np.array converts the selected columns into an array. The drop
# function returns the data frame without the specified column. Our X array
# now contains all of our columns, except for the final grade. The final grade is
# in the Y array.
X_train,X_test,Y_train,Y_test, = train_test_split(X,Y,test_size=0.1)
# The second part(test_size) then checks the accuracy of the prediction, with previously
# unknown data.

# With the function train_test_split , we divide our X and Y arrays into four
# arrays. The order must be exactly as shown here. The test_size parameter
# specifies what percentage of records to use for testing. In this case, it is 10%.
# This is also a good and recommended value. We do this to test how accurate
# it is with data that our model has never seen before.
model = LinearRegression()
model.fit(X_train, Y_train)
# We then use the fit function and pass our training data. Now our model is
# already trained. It has now adjusted its hyperplane so that it fits all of our
# values.

accuracy = model.score(X_test, Y_test)
# In order to test how well our model performs, we can use the score method
# and pass our testing data.
print('{}%'.format(accuracy*100))

# Actually, 85 percent is a pretty high and good accuracy. Now that we know
# that our model is somewhat reliable, we can enter new data and predict the
# final grade.

X_new = np.array([[18,1,3,40,15,16]])
Y_new = model.predict(X_new)
print(Y_new)
# We use the predict method, to calculate the likely final grade for
# our inputs.
# The final grade would probably be 17.

# Visualizing Correlations
# However, what we can visualize are relationships between individual features.
# Here we draw a scatter plot with the function scatter, which shows the
# relationship between the learning time and the final grade.
# plt.scatter(data['studytime'], data['G3'])
# plt.title('Correlation')
# plt.xlabel('Study Time')
# plt.ylabel('Final Grade')

# if we look at the correlation between the second grade and the final
# grade, we see a much stronger correlation.
# Here we can clearly see that the students with good second grades are very
# likely to end up with a good final grade as well.
plt.scatter(data['G2'], data['G3'])
plt.title('Correlation')
plt.xlabel('Second Grade')
plt.ylabel('Final Grade')

plt.show() 
