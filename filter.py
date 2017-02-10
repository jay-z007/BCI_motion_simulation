import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from helperFunctions import *
from math import log
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, '../dataset', 'dataset3.5', 'subject1_ascii')
data_to_be_referred = [7,8,11,12,18,21,22,31]
files = ['train_subject1_raw01.asc']#, 'train_subject1_raw02.asc', 'train_subject1_raw03.asc']
fs_Hz = 512.0   # assumed sample rate for the EEG data
NFFT = 512      # pick the length of the fft
overlap = NFFT - 50  # fixed step of 50 points

f_res_2 = []
pxx_res_2 = []
data = []
pxx_temp = []
counter = 0
avg = np.zeros(96)
eeg_data_uV = []
eeg_temp = []
target = []

# fs = 5000.0
# lowcut = 500.0
# highcut = 1250.0

# T = 0.05
# nsamples = T * fs
# t = np.linspace(0, T, nsamples, endpoint=False)
# a = 0.02
# f0 = 600.0
# x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
# x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
# x += a * np.cos(2 * np.pi * f0 * t + .11)
# x += 0.03 * np.cos(2 * np.pi * 2000 * t)
# plt.figure(2)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')

# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
# plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
# plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')

# plt.show()

#for file in files:
temp_eeg_data_uV, temp_target = loadData(os.path.join(data_dir, files[0]), fs_Hz)
eeg_data_uV.extend(temp_eeg_data_uV)
target.extend(temp_target)	

#t_sec = (np.arange(f_eeg_data_uV.size))/fs_Hz

# eeg_temp = eeg_data_uV[1024:1536]
# eeg_temp = np.transpose(eeg_temp)
# eeg_temp_sel = eeg_temp[2]

# for row in eeg_temp:
# 	print "\n\n", row

# fft_plot(eeg_temp_sel)

# for i in range(32):

# 	# plt.subplot(2, 1, 1)
# 	# plt.xlim([512, 1024])
# 	# plt.plot(eeg_temp_sel)
# 	# print eeg_temp[2]
# 	#plt.show()

# f_eeg_temp = butter_bandpass_filter(eeg_temp_sel, 8, fs_Hz,'high')
# f_eeg_temp = butter_bandpass_filter(f_eeg_temp, 30, fs_Hz,'low')
# f_eeg_temp = butter_bandpass_filter(f_eeg_temp, 30, fs_Hz,'low')
# fft_plot(f_eeg_temp)
	# plt.subplot(2, 1, 2)
	# plt.xlim([0, 512])
	#plt.ylim([5000, 7000])
	# plt.plot(f_eeg_temp)
	#plt.show()

	# plt.figure(figsize=(9, 6))

	# f, pxx = signal.welch(f_eeg_temp, fs_Hz, nperseg=512, noverlap=overlap)
	# plt.plot(f, pxx)
	# plt.ylim([0, 100])
	# plt.xlabel("Frequency (Hz)")
	# plt.ylabel("PSD (dB/Hz) %g"%i)
	# plt.show()


#*********************************************************
print len(target)
counter = 0
for i in range(len(eeg_data_uV)/512):
	one_sec_eeg_data_uV = eeg_data_uV[i*512:(i+1)*512]
	# print "****",len(one_sec_eeg_data_uV[1]),'\n\n\n'
	one_sec_eeg_data_uV = np.transpose(one_sec_eeg_data_uV)

	for row in range(len(one_sec_eeg_data_uV)):
		counter += 1
		f_eeg_data_uV = butter_bandpass_filter(one_sec_eeg_data_uV[row], 8, fs_Hz,'high')
		f_eeg_data_uV = butter_bandpass_filter(f_eeg_data_uV, 30, fs_Hz,'low')
		f_eeg_data_uV = butter_bandpass_filter(f_eeg_data_uV, 30, fs_Hz,'low')

		f, pxx = signal.welch(f_eeg_data_uV, fs_Hz, nperseg=512, noverlap=overlap)

		pxx_res_2 = pxx[8:31:2]
		# f_res_2.append(f[8:31:2])
		pxx_temp.extend(pxx[8:31:2])
		# print '*********',len(pxx[8:31:2])
		#print pxx_res_2

		# print counter,i
		# plt.xlabel(counter)
		# plt.plot(f[8:31:2], pxx[8:31:2],'r')
		# plt.plot(f, pxx,'g')
		# plt.ylim([0,1000])
		# plt.show()

	data.append(pxx_temp)
	# 	counter = 0
	
	pxx_temp = [] 	
print data[0]
print counter
#**************************************
# for i in range(len(data)):
# 	print data[i],'\n'
# print 'lendata',len(data),'lentarget',len(target)
# print f_res_2
# print '------------------',len(pxx_res_2),'\n\n'
# for i in range(len(data)):
# 	# print pxx_res_2[i],'\n\n'
# 	for j in range(len(data[i])) :
# 		data[i][j] = log(data[i][j])
	# print pxx_res_2[i],'\n\n'
# print pxx_res_2
# full_spec_PSDperBin, full_t_spec, freqs = convertToFreqDomain(f_eeg_data_uV, fs_Hz, NFFT, overlap)
# spec_PSDperBin = full_spec_PSDperBin[:, 1:-1:2]  # get every other time slice
# t_spec = full_t_spec[1:-1:2]  # get every other time slice

# make the spectrogram plot
#plt.pcolor(t_spec, freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
# plt.clim(25-5+np.array([-40, 0]))
# plt.xlim(t_sec[0], t_sec[-1])
# if (t_lim_sec[2-1] != 0):
#     plt.xlim(t_lim_sec)
# plt.ylim(f_lim_Hz)
# plt.xlabel('Time (sec)')
# plt.ylabel('Frequency (Hz)')
#plt.title(fname[12:])
