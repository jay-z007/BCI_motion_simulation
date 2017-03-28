import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from helperFunctions import *

root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, '../dataset', 'dataset3.5')
# subject_arr = ["subject1_ascii","subject2_ascii","subject3_ascii"]

files = [
'subject1_ascii/train_subject1_raw1.asc'
  , 'subject1_ascii/train_subject1_raw2.asc'
 # , 'subject1_ascii/train_subject1_raw3.asc'

  , 'subject2_ascii/train_subject2_raw1.asc'
 #, 'subject2_ascii/train_subject2_raw2.asc'
 # , 'subject2_ascii/train_subject2_raw3.asc'

  , 'subject3_ascii/train_subject3_raw1.asc'
 # , 'subject3_ascii/train_subject3_raw2.asc'
 # , 'subject3_ascii/train_subject3_raw3.asc'
]

fs_Hz = 512.0   # assumed sample rate for the EEG data
# NFFT = 512      # pick the length of the fft

# f_res_2 = []
# pxx_res_2 = []

data = []
# pxx_temp = []
# avg = np.zeros(96)
eeg_data_uV = []
eeg_temp = []
target = []
section_size = 64
new_data = []
for file in files:
	temp_eeg_data_uV, temp_target = loadData(os.path.join(data_dir, file), fs_Hz)
	eeg_data_uV.extend(temp_eeg_data_uV)
	target.extend(temp_target)	
target = target[::section_size]

data = eeg_data_uV
# print "len of data0--",len(data[0]),data[0]
# data = np.array(data)
#*********************************************************
# print len(eeg_data_uV),len(target)
counter = 0
count = 0

print len(data), len(data[0])

for i in range(len(data)/section_size):
	data_part = np.array(data[i*section_size:(i+1)*section_size])
# data = np.transpose(data)
# for i in range()
	count +=1
	data_part = data_part.T
	# new_target = target[i*8]
# print len(data)
# for row in data[0]:
# 	print row

	# print count
	for row in range(len(data_part)):
		counter += 1
		#plt.plot(data[row], 'r')

#		f, pxx = signal.welch(data_part[row], fs_Hz, nperseg=512)

#		plt.plot(f, pxx, 'r')

		data_part[row] = butter_bandpass_filter(data_part[row], 8, 30,fs_Hz,'band')

		f, pxx = signal.welch(data_part[row], fs_Hz, nperseg=64)

#		plt.plot(f, pxx, 'g')
#		plt.show()
		
		eeg_temp.extend(pxx[8:31])
		#data_part = data_part.T
	#new_data.append(eeg_temp)
	new_data.append(eeg_temp)
	eeg_temp = []

# print new_data
data = new_data

print len(data), len(data[0])
# target = new_target
# print "length of data",len(data), "target",len(target)
# print "counter",counter
# print data
# data = f_eeg_data_uV
# print counter
#**************************************
