import random
from numpy import *
from neural_network import *
from helper_functions import *
from init import *
import signals

X = signals.features
Y = signals.target
X_len = len(X)
Y_len = len(Y)

print X_len, Y_len

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2)
len_Y_train = len(Y_train)
len_X_train = len(X_train)

print '\n\n\n\n',len_Y_train,len_X_train,'\n\n\n\n\n'
print '\n\n\n\n',Y_train[0],'\n\n\n',len(X[0]),X[20],'\n\n\n\n\n'


for i in range(len_Y_train):
	temp = [0]*4
	temp[Y_train[i]-1] = 1
	Y_train[i] = temp

# print "\n\nX = ", X, 
print "\n\nY = ", Y
cnt = 0
for i in range(len(X)):
	for j in range(len(X[i])):
		if not isfinite(X[i][j]):
			print X[i][j]
			cnt += 1
			print cnt, i ,j 


# for i in range(len(X)):
# 	for j in range(len(X[i])):
# 		if isnan(X[i][j]):
# 			print '***',X[i][j]
# 			cnt += 1
# 			print cnt, i ,j 



from sklearn.neural_network import MLPClassifier
my_classifier = MLPClassifier(hidden_layer_sizes=(200), activation='logistic', alpha=0.00005, #learning_rate='invscaling',
					max_iter=50, tol=0.00000001, verbose=True, warm_start=True)

for i in range(20):
	print '---',i
	my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

# for i in range(len(predictions)):
# 	predictions[i]		

for i in range(len(Y_test)):
	print predictions[i], Y_test[i]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print accuracy

##########################
