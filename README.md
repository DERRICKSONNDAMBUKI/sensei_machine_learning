## Machine Learning

we focus on teaching machines or computers to perform certain tasks without being given specific instructions.

Artificial intelligence is a broad field and every system that can learn and solve problems might be considered an AI. Machine learning is one specific approach to this broad field. In machine learning the AI doesn’t receive any instructions.

It’s dynamic and restructures itself.

The two main approaches are supervised learning and unsupervised learning.
Others are: reinforcement learning.

#### Supervised Learning

Here, we give our model a set of inputs and also the corresponding outputs, which are the desired results. In this way, the model learns to match certain inputs to certain outputs and it adjusts its structure. It learns to make connections between what was put in and what the desired output is. It understands the correlation. When trained well enough, we can use the model to make predictions for inputs that we don’t know the results for.

Classic supervised learning algorithms are regressions classifications and support vector machines.

#### Unsupervised Learning

With unsupervised learning on the other hand, we don’t give our model the desired results while training. Not because we don’t want to but because we don’t know them. This approach is more like a kind of pattern recognition. We give our model a set of input data and it then has to look for patterns in it.
Once the model is trained, we can put in new data and our model will need to make decisions.
Since the model doesn’t get any information about classes or results, it has to work with similarities and patterns in the data and categorize or cluster it by itself.
Classic unsupervised learning algorithms are clustering, anomaly detection and some applications of neural networks.

#### Reindforcement Learning

Here we create some model with a random structure. Then we just observe what it does and reinforce or encourage it, when we like what it does. Otherwise, we can also give some negative feedback. The more our model does what we want it to do, the more we reinforce it and the more “rewards” it gets. This might happen in form of a number or a grade, which represents the so-called fitness of the model. 
In this way, our model learns what is right and what is wrong. You can imagine it a little bit like natural selection and survival of the fittest. We can create 100 random models and kill the 50 models that perform worst. Then the remaining 50 reproduce and the same process repeats. These kinds of algorithms are called genetic algorithms. 
Classic reinforcement learning algorithms are genetic or evolutional algorithms.

## Deep Learning

Deep learning however is just one area of machine learning, namely the one, which works with neural networks. Neural networks are a very comprehensive and complex topic.

#### Fields Of Machine Learning Application

- Research
- Autonomous Cars
- Spacecraft
- Economics and Finance
- Medical and Healthcare
- Physics, Biology, Chemistry
- Engineering
- Mathematics
- Robotics
- Education
- Forensics
- Police and Military
- Marketing
- Search Engines
- GPS and Pathfinding Systems
- ...

##### Scikit-learn

It features classification, regression and clustering algorithms.

##### Tensorflow

It is a whole ecosystem for developing modern deep learning models. This means that it is mainly used for the development and training of models that use neural networks. It also has its own data structures and ways of visualizing data.

#### Modules to install

pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install tensorflow

#### 1 Linear Regression

The easiest and most basic machine learning algorithm is linear regression.
It is a supervised learning algorithm. That means that we need both – inputs and outputs – to train the model.
The x-value is called the feature , whereas the y-value is our
label . The label is the result for our feature. Our linear regression model is represented by the blue line that goes straight through our data. It is placed so that it is as close as possible to all points at the same time. So we “trained” the line to fit the existing points or the existing data.

### Classification Algorithms
#### K-Nearest-Neighbours

we assign the class of the new object, based on its nearest neighbors. The K specifies the amount of neighbors to look at. Notice that K shouldn’t be a multiple of the number of classes since it might cause conflicts when we have an equal amount of elements from one class as from the other.

#### Naive-Bayes

#### Logistic Regression
It looks at probabilities and determines how likely it is that a certain event
happens (or a certain class is the right one), given the input data. This is done
by plotting something similar to a logistic growth curve and splitting the data
into two.

#### Decision Trees

#### Random Forest
It is based on decision trees. What it does is creating a forest of multiple
decision trees. To classify a new object, all the various trees determine a class
and the most frequent result gets chosen. This makes the result more accurate
and it also prevents overfitting. It is also more suited to handle data sets with
higher dimensions. On the other hand, since the generation of the forest is
random , you have very little control over your model.

# PREDICTING LABELS
Again, we can again make predictions for new, unknown data. The chance of success in the classification is even very high. We just need to pass an array of input values and use the predict function.

X_new = np.array([[...]])
Y_new = clf.predict(X_new)

## Support Vector Machines

These are very powerful, very efficient machine learning algorithms and they even achieve much better results than neural networks in some areas.

#### kernels
 These add a new dimension to our data. By doing that, we hope to increase the complexity of the data and possibly use a hyperplane as a separator.

#### soft margin
Now instead of using a kernel or a polynomial function to solve this problem, we can define a so-called soft margin. With this, we allow for conscious misclassification of outliers in order to create a more accurate model. Caring too much about these outliers would again mean overfitting the model.

# 2. Unsupervised Learning Algorithm
## Clustering

Clustering is an unsupervised learning algorithm, which means that we don’t have the results for our inputs. We can’t tell our model what is right and wrong. It has to find patterns on its own.
The algorithm gets raw data and tries to divide it up into clusters. K-Means-Clustering is the method that we are going to use here. Similar to K-Nearest-Neighbors, the K states the amount of clusters we want.
The clustering itself works with so-called centroids . These are the points, which lie in the center of the respective clusters.

# Neural Networks
- - - 
## Optimizing Models
 - Serialization - used to save objects into files during       runtime. 
 By doing this, we are not only saving the attributes
 but the whole state of the object. Because of that, we can load the same object back into a program later and continue working with it.
 






AI provide the capabilities to:
- Generalize/adapt
- Reason
- Solve problem

Types of ai:
* Weak AI - perform solely on one task
* Strong AI (exist fiction)
