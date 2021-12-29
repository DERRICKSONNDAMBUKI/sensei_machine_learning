from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
# print(data.feature_names)
# print(data.target_names)
# Our features are all parameters that should help to determine the label or the target. For the targets, we have two options in this
# dataset: malignant and benign .

# Preparing data
X=np.array(data.data)
Y=np.array(data.target)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1) # test_size = 10%


clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = GaussianNB()
clf3 = LogisticRegression()
clf4 = DecisionTreeClassifier()
clf5 = RandomForestClassifier()

clf1.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)
clf3.fit(X_train, Y_train)
clf4.fit(X_train, Y_train)
clf5.fit(X_train, Y_train)

print('Knearest-neighbor accuracy is {}%'.format(clf1.score(X_test, Y_test)*100))
print('Naive bayes accuracy is {}%'.format(clf2.score(X_test, Y_test)*100))
print('Logistic Regression accuracy is {}%'.format(clf3.score(X_test, Y_test)*100))
print('Decision tree accuracy is {}%'.format(clf4.score(X_test, Y_test)*100))
print('Random forest accuracy si {}%'.format(clf5.score(X_test, Y_test)*100))

# PREDICTING LABELS
# Again, we can again make predictions for new, unknown data. The chance of
# success in the classification is even very high. We just need to pass an array
# of input values and use the predict function.
# 
# X_new = np.array([[...]])
# Y_new = clf.predict(X_new)