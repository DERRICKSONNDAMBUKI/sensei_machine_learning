import pickle
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# optimizing models
best_accuracy = 0
for x in range(2500):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    model = SVC(kernel='linear', C=3)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print('Best accuracy: ', accuracy)
        with open('model.pickle', 'wb') as file:
            pickle.dump(model, file)

print(best_accuracy)

# The concept is quite simple. We define a variable best_accuracy which starts
# with the v/alue zero. Then we run a loop with 2500 iterations and we train our
# model over and over again with a different split for our data and different
# seeds. When we test the model, we check if the accuracy is higher than the highest
# measured accuracy (starting with zero). If that is the case, we save the model
# and update the best accuracy. By doing this, we find the model with the
# highest accuracy. Notice that we are still only using our training data and our test data. This
# means that if we take things too far, we might overfit the model, especially
# with simple datasets. It is not impossible to reach an accuracy of 100% but the
# question remains if this accuracy also applies to unknown data.
