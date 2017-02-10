import random
from numpy import *
#from neural_network import *
from helperFunctions import *
#from init import *
import filter

X = filter.data
Y = filter.target
X_len = len(X)
print 'lenX',len(X),'lenY',len(Y)

from sklearn.preprocessing import scale
X = scale(X)

# print X[0]

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2)
len_Y_train = len(Y_train)

# print "\n\nX = ", X, "\n\nY = ", Y

from sklearn.neural_network import MLPClassifier
my_classifier = MLPClassifier(hidden_layer_sizes=(60), activation='logistic', alpha=0.0001, learning_rate='invscaling',
					max_iter=20, tol=0.00000001, verbose=True, warm_start=True)

# from sklearn.cluster import KMeans
# my_classifier = KMeans(n_clusters=2, n_init=5, n_jobs=-1)


for i in range(300):
	print i
	my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

for i in range(len(Y_test)):
	print predictions[i], Y_test[i]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print accuracy

##########################	