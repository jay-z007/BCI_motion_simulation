import os.path
import numpy as np
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

import matplotlib.pyplot as plt
import pywt

data = []
target = []
trig = []

db20 = pywt.Wavelet('db20')

coeffs = []
new_coeffs = []
wave2 = []
features = []
#low to high
coeffs_to_extract = [['a',6], ['d', 6], ['d', 5], ['d', 4], ['d', 3]]
# color = ['b', 'g', 'r', 'c', 'm', 'k', '#4527a0','#009688','#ff9800']


def get_name(n, case):
	prefix = "../dataset/dataset3.3a/set"
	suffix = ""
	
	if case == 1:
		suffix = "/temp.txt"
	elif case == 2:
		suffix = "/HDR_Classlabel.txt"
	elif case == 3:
		suffix = "/HDR_TRIG.txt"

	fname = prefix+str(n)+suffix
	return fname

####################
#
#	Extract required coefficients from the original wave coefficients
#	coeffs_to_extract ==> is a 2D list with each row having 2 elements
#							1. the type of coeff ['a' or 'd']
#							2. level of the coeff [1, 2, .. N]
#
####################
def extract_coeffs(original_coeffs, coeffs_to_extract):
	indices = []
	new_coeffs = []
	clen = len(original_coeffs)

	for row in coeffs_to_extract:
		if row[0] == 'a':
			indices.append(0)
		elif row[0] == 'd':
			indices.append(clen-row[1])
	
	for i in range(clen):
		if i in indices:
			new_coeffs.append(original_coeffs[i])
		else:
			new_coeffs.append(np.zeros(len(original_coeffs[i])))

	return new_coeffs


def plotSpectrum(y,Fs,color):
	"""
	Plots a Single-Sided Amplitude Spectrum of y(t)
	"""
	n = len(y) # length of the signal
	#k = np.fft.fftfreq(n, d=0.005)
	k = np.arange(0, n/2, 1)
	T = n/Fs
	#frq = k/T # two sides frequency range
	#frq = frq[range(n/2)] # one side frequency range
	#print "k = ", k

	Y = np.fft.fft(y)#/n # fft computing and normalization
	Y = Y[range(n/2)]
	mag_y = abs(Y)
	max_mag_y = max(mag_y)

	id = np.where(mag_y==max_mag_y)
	#print id, max_mag_y, mag_y

	plot(k, abs(Y), color) # plotting the spectrum
	xlabel('Freq (Hz)')
	ylabel('|Y(freq)|')
#	show()

def extract_features(wave):
	return [np.nanmean(wave), np.nanstd(wave), np.nanmin(wave), np.nanmax(wave)]
	#return feature_vector

fname_data = get_name(3, 1)
fname_target = get_name(3, 2)
fname_trail_trig = get_name(3, 3)
data_temp = []
features = []
# for i in range(1):
counter = 0
try:
	with open(fname_data) as data_file, open(fname_target) as target_file, \
	open(fname_trail_trig) as fname_trail_trig:

		# counter_NaN = 0
		target_line = ""
		trig_line = fname_trail_trig.readline()		
		trig_flag, data_flag = True, True
		while True:
			counter += 1

			data_line = data_file.readline()

			if data_line == "" and data_flag:
				print "\nData file end\n"
				data_flag = False

			if not trig_flag and not data_flag:
				break

			arr = data_line.split()
			for j in range(len(arr)):
				arr[j] = float(arr[j])
			data_temp.append(arr)

			if trig_flag and counter == int(trig_line):	
				
				target_line = target_file.readline()
				trig_line = fname_trail_trig.readline()

				if target_line == "NaN\n":
					pass
				elif trig_line == "" and trig_flag:
					print "\nTrig file end\n"
					trig_flag = False
				else:
					target.append(int(target_line))
					trig.append(int(trig_line))			
					data.append(data_temp)
				data_temp = []


			
except IOError as e:
	print 'Operation failed: %s' % e.strerror

counter_data = 0 
counter_new_matrix = 0 
counter_wave2 = 0 
# print len(data)


for matrix in data:
	counter_data += 1
	# print counter_data
	new_matrx = np.transpose(matrix)
	
	counter_new_matrix = 0
	temp_features = []
		
	for new_matrix in new_matrx:
		counter_new_matrix += 1
		#print 'length of new_matrix',len(new_matrix)
		coeffs = pywt.wavedec(new_matrix, db20)
		new_coeffs = []
		wave2 = []
		for row in coeffs_to_extract:
			row = [row]
			new_coeffs.append(extract_coeffs(coeffs, row))
		for row in new_coeffs:
			wave2.append(pywt.waverec(row, db20))
		counter_wave2 = 0
		#print 'len of wave2',len(wave2)
		for row in wave2:
			counter_wave2 += 1
			temp_features.extend(extract_features(row))
	features.append(temp_features)
