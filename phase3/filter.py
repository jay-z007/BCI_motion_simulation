import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from helperFunctions import *

root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, '../dataset', 'dataset3.5')
subject_arr = ["subject1_ascii","subject2_ascii","subject3_ascii"]

files = [
'subject1_ascii/train_subject1_raw1.asc', 'subject1_ascii/train_subject1_raw2.asc', 'subject1_ascii/train_subject1_raw3.asc'
# 'subject2_ascii/train_subject2_raw1.asc', 'subject2_ascii/train_subject2_raw2.asc', 'subject2_ascii/train_subject2_raw3.asc',
# 'subject3_ascii/train_subject3_raw1.asc', 'subject3_ascii/train_subject3_raw2.asc', 'subject3_ascii/train_subject3_raw3.asc'
]

fs_Hz = 512.0   # assumed sample rate for the EEG data
NFFT = 512      # pick the length of the fft

f_res_2 = []
pxx_res_2 = []
data = []
pxx_temp = []
avg = np.zeros(96)
eeg_data_uV = []
eeg_temp = []
target = []

for file in files:
	temp_eeg_data_uV, temp_target = loadData(os.path.join(data_dir, file), fs_Hz)
	eeg_data_uV.extend(temp_eeg_data_uV)
	target.extend(temp_target)	

data = eeg_data_uV

# #*********************************************************

# #counter = 0

# for i in range(len(eeg_data_uV)/512):
# 	one_sec_eeg_data_uV = eeg_data_uV[i*512:(i+1)*512]
# 	one_sec_eeg_data_uV = np.transpose(one_sec_eeg_data_uV)


# 	for row in range(len(one_sec_eeg_data_uV)):
# 		#counter += 1
# 		f_eeg_data_uV = butter_bandpass_filter(one_sec_eeg_data_uV[row], 8, fs_Hz,'high')
# 		f_eeg_data_uV = butter_bandpass_filter(f_eeg_data_uV, 30, fs_Hz,'low')

# 		f, pxx = signal.welch(f_eeg_data_uV, fs_Hz, nperseg=512)

# 		pxx_res_2 = pxx[8:31:2]
# 		pxx_temp.extend(pxx_res_2)

# 		# print counter,i
# 		# plt.xlabel(counter)
# 		# plt.plot(f[8:31:2], pxx[8:31:2],'r')
# 		# plt.plot(f, pxx,'g')
# 		# plt.ylim([0,1000])
# 		# plt.show()
	
# 	data.append(pxx_temp)	
# 	pxx_temp = []

# # print counter
# #**************************************
