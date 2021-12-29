from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
Y= data.target 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)
model = SVC(kernel='linear',C= 3)
model.fit(X_train, Y_train)

with open('model.pickle','rb') as file:
    model = pickle.load(file)
    # Here we open a file stream in the read bytes mode and use the load function
    # of the pickle module to get the model into our program.
    
model.predict([...])
