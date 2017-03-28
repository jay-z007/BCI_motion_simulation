import os
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import accuracy_score
import tensorflow as tf
import time

start_time = time.time()
import filter_part


def dense_to_one_hot(labels_dense, num_classes=2):
	"""Convert class labels from scalars to one-hot vectors"""
	num_labels = labels_dense.shape[0]
	#index_offset = np.arange(num_labels)# * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	#labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

	index = 0
	for i in range(num_labels):
		if labels_dense[i] == 5:
			index = 2
		elif labels_dense[i] == 0:
			index = 0
		else:
			index = 1

		labels_one_hot[i][index] = 1

	return labels_one_hot

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('../')
data_dir = os.path.join(root_dir, 'dataset', 'dataset3.5', 'data_psd')
#sub_dir = os.path.join(root_dir, 'sub')

print 'check for existence'
print 'root_dir -', os.path.exists(root_dir)
print 'data_dir -', os.path.exists(data_dir)
#os.path.exists(sub_dir)

# #load the csv as pd dataframes
# train = pd.read_csv(os.path.join(data_dir, 'train_subject1_psd01.asc'), delim_whitespace=True)
# #test = pd.read_csv(os.path.join(data_dir, 'test_subject1_psd04.asc'), delim_whitespace=True)

#convert dataframes to matrices
# train_x = train.ix[:, 0:96].as_matrix()
# train_y = dense_to_one_hot(train.ix[:, 96:].as_matrix().flatten())
# #test_x = test.as_matrix()

# print train_x
# print train_y

train_x = pd.DataFrame(filter_part.data).as_matrix()
train_y = dense_to_one_hot (pd.DataFrame(filter_part.target).as_matrix().flatten())

print len(train_x), len(train_x[0])

from sklearn.decomposition import PCA
pca = PCA(n_components=16)
train_x = pca.fit_transform(train_x)
# n_components=8

# print pca.explained_variance_ratio_
# print train_x

from sklearn.cross_validation import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = .2)
print train_y

# #split into train and validation sets
# split_size = int(train_x.shape[0]*0.80)
# train_x, val_x = train_x[:split_size], train_x[split_size:]
# train_y, val_y = train_y[:split_size], train_y[split_size:]
print "Training start"
# print "\n\n",train_x
# print "\n\n",val_x
# print "\n\n",train_y
# print "\n\n",val_y

# for i in train_y:
# 	print i
### set all variables

# number of neurons in each layer
# input_num_units = n_components
# hidden_num_units = 40
# output_num_units = 2

###############################################
from sklearn.neural_network import MLPClassifier
my_classifier = MLPClassifier(hidden_layer_sizes=(30), activation='logistic', alpha=0.0001, learning_rate='invscaling',
					max_iter=20, tol=0.00000001, verbose=True, warm_start=True)

for i in range(400):
	print '---',i
	my_classifier.fit(train_x, train_y)

predictions = my_classifier.predict(val_x)

# for i in range(len(predictions)):
# 	predictions[i]		

for i in range(len(val_y)):
	print predictions[i], val_y[i]
# 	predictions[i] = np.where(predictions[i] == 1)
# 	val_y[i] = np.where(val_y[i] == 1)	

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(val_y, predictions)
print accuracy

##########################

# # define placeholders
# x = tf.placeholder(tf.float32, [None, input_num_units])
# y = tf.placeholder(tf.float32, [None, output_num_units])

# # set remaining variables
# epochs = 300
# batch_size = 128
# learning_rate = 0.0001


# ### define weights and biases of the neural network

# weights = {
# 	'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
# 	'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
# }

# biases = {
# 	'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
# 	'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
# }

# hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
# hidden_layer = tf.nn.sigmoid(hidden_layer)

# output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init = tf.global_variables_initializer()

# #with tf.Session() as sess:
# sess = tf.Session()
# # create initialized variables
# sess.run(init)

# for epoch in range(epochs):
# # avg_cost = 0
# # total_batch = int(train.shape[0]/batch_size)
# 	#print "\nEpoch : ",epoch
# 	c = sess.run([optimizer, cost], feed_dict = {x: train_x.reshape(-1, input_num_units), y: train_y})
# 	#print "cost : ",c

# #avg_cost += c / total_batch
#  #   print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)

# print "\nTraining complete!"


# # find predictions on val set
# pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

# acc = sess.run(accuracy, feed_dict={x: val_x.reshape(-1, input_num_units), y: val_y})

# print "Validation Accuracy:", acc*100,"%" 
# print("--- %s seconds ---" % (time.time() - start_time))

# #printacc(acc)

# # predict = tf.argmax(output_layer, 1)
# # pred = sess.run(predict, feed_dict={x: test_x.reshape(-1, input_num_units)})

# #print pred







# def preproc(unclean_batch_x):
# 	"""Convert values to range 0-1"""
# 	temp_batch = unclean_batch_x / unclean_batch_x.max()

# 	return temp_batch

# def batch_creator(batch_size, dataset_length, dataset_name):
# 	"""Create batch with random samples and return appropriate format"""
# 	batch_mask = rng.choice(dataset_length, batch_size)

# 	batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, 96)
# 	batch_x = preproc(batch_x)

# 	if dataset_name == 'train':
# 		batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
# 		batch_y = dense_to_one_hot(batch_y)

# 	return batch_x, batch_y
