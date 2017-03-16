import random
from numpy import *
from Tkinter import *
from helperFunctions import *
import ball_simulation
import filter
import time

X = filter.data
Y = filter.target
X_len = len(X)
print 'lenX',len(X),'lenY',len(Y)


from sklearn.preprocessing import scale
X = scale(X)
# from sklearn.preprocessing import normalize
# X = normalize(X)
#print '*******************************************************',X[0],'****************************************************'
from sklearn.decomposition import PCA
pca = PCA(n_components=8)

X = pca.fit_transform(X)
# print '///////////////////////',X[0],'//////////////////////'
# print '/////',len(X)
# print '*****',len(X[0])

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2)
len_Y_train = len(Y_train)

from sklearn.neural_network import MLPClassifier
my_classifier = MLPClassifier(hidden_layer_sizes=(20), activation='logistic', alpha=0.0001, learning_rate='invscaling',
					max_iter=20, tol=0.00000001, verbose=True, warm_start=True)

# from sklearn.cluster import KMeans
# my_classifier = KMeans(n_clusters=2, n_init=10, n_jobs=-1)

# from sklearn.svm import SVC
# my_classifier = SVC()

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

for i in range(200):
	print i
	my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

##### Print prediction and move the ball in the simulation

# initialize root Window and canvas
# root = Tk()
# root.title("Balls")
# root.resizable(False,False)
# canvas = Canvas(root, width = 700, height = 700)
# canvas.pack()

# # create two ball objects and animate them
# ball1 = ball_simulation.Ball(canvas, 300, 300, 350, 350)

for i in range(len(Y_test)):
	# root.update()
	# if predictions[i] == Y_test[i]:
	print predictions[i], Y_test[i]#, (predictions[i]-0.5)*10
	# else:
	# 	print 'no'
	# # ball1.move_ball((predictions[i]-0.5)*50)
	# time.sleep(.200)

	#root.mainloop()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print accuracy

##########################	