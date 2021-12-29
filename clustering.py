from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

digits = load_digits()
data = scale(digits.data)
print(data)
# After loading our dataset we use the scale function, to standardize our data.
# We are dealing with quite large values here and by scaling them down to
# smaller values we save computation time.


# training predicting
clf = KMeans(n_clusters=10,init='random',n_init=10)
clf.fit(data)
# Here we are passing three parameters. The first one (n_clusters ) defines the
# amount of clusters we want to have. Since we are dealing with the digits 0 to
# 9, we create ten different clusters.
# With the init parameter we choose the way of initialization. Here we chose
# random , which obviously means that we just randomly place the centroids
# somewhere. Alternatively, we could use k-means++ for intelligent placing.
# The last parameter (n_init ) states how many times the algorithm will be run
# with different centroid seeds to find the best clusters.
# Since we are dealing with unsupervised learning here, scoring the model is
# not really possible. You won’t be able to really score if the model is clustering
# right or not. We could only benchmark certain statistics like completeness or
# homogeneity .
# clf.predict([X])
