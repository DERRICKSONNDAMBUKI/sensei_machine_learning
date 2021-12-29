from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# This is the support vector classifier that we are going to use as our model.
# Notice that we are also importing the KNeighborsClassifier again, since we
# are going to compare the accuracies at the end.

data = load_breast_cancer()
X=data.data
Y= data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=30)
# This time we use a new parameter named random_state . It is a seed that
# always produces the exact same split of our data. Usually, the data gets split
# randomly every time we run the script. You can use whatever number you
# want here. Each number creates a certain split which doesn’t change no
# matter how many times we run the script. We do this in order to be able to
# objectively compare the different classifiers.

# training and testing
model = SVC(kernel='linear',C=3)
model.fit(X_train, Y_train)
# The first one is our kernel and the second one is C which is our soft margin. Here
# we choose a linear kernel and allow for three misclassifications. Alternatively
# we could choose poly, rbf, sigmoid, precomputed or a self-defined kernel.
# Some are more effective in certain situations but also a lot more time-
# intensive than linear kernels.
accuracy = model.score(X_test, Y_test)
print('SVC accuracy is {}%'.format(accuracy*100))

# KNeighborsClassifier with the same random_state
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

knn_accuracy = knn.score(X_test, Y_test)
print('KNN accurary is {}%'.format(accuracy*100))


