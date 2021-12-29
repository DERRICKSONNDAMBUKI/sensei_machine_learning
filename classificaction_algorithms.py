import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer # data on breats cancer

data = load_breast_cancer()
# print(data.feature_names)
# print(data.target_names)
# Our features are all parameters that should help to determine the label or the target. For the targets, we have two options in this
# dataset: malignant and benign .

# Preparing data
X=np.array(data.data)
Y=np.array(data.target)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1) # test_size = 10%

# training and testing
knn = KNeighborsClassifier(n_neighbors=5) # 5 neighbor points that we want to consider
knn.fit(X_train, Y_train)
accuracy = knn.score(X_test, Y_test)
print('{}%'.format(accuracy*100))


