import matplotlib.pyplot as plt
import pywt
import numpy as np
# import time
import os.path
import signals

f_lim_Hz = [2000, 5000]

data = signals.data
data = data[:2]
tar = signals.target

db6 = pywt.Wavelet('db20')

coeffs = []
new_coeffs = []
wave2 = []
features = []
coeffs_to_extract = [['a',6], ['d', 6], ['d', 5], ['d', 4], ['d', 3]]
color = ['b', 'g', 'r', 'c', 'm', 'k', '#4527a0','#009688','#ff9800']


counter_data = 0 
counter_new_matrix = 0 
counter_wave2 = 0 
for matrix in data:
	counter_data += 1
	new_matrx = np.transpose(matrix)
	features = []

	counter_new_matrix = 0
	for new_matrix in new_matrx:
		counter_new_matrix += 1
		print 'length of new_matrix',len(new_matrix)
		coeffs = pywt.wavedec(new_matrix, db6)
		#low to high
		# print 'coeffs',len(coeffs)

		new_coeffs = []
		wave2 = []

		for row in coeffs_to_extract:
			row = [row]
			new_coeffs.append(signals.extract_coeffs(coeffs, row))

		for row in new_coeffs:
			wave2.append(pywt.waverec(row, db6))
		
		counter_wave2 = 0
		print 'len of wave2',len(wave2)
		for row in wave2:
			counter_wave2 += 1
			temp = signals.extract_features(row)
			# print temp , counter_data , counter_new_matrix, counter_wave2
			features.append(temp)
			# row = [row[i]+counter*100 for i in range(len(row))]
			# print "\n\n#################################",counter, "\n\n",row
			#plt.setp(plt.plot(row), color=color[counter])
			
			# counter += 1
	print len(features)

