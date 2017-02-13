import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack

#data_to_be_referred = [7,8,11,12,18,21,22,31]
electrode_map = {
    "Fp1":0,  "AF3":1,
    "F7":2,   "F3":3,
    "FC1":4,  "FC5":5,
    "T7":6,   "C3":7,
    "CP1":8,  "CP5":9,
    "P7":10,  "P3":11,
    "Pz":12,  "PO3":13,
    "O1":14,  "Oz":15,
    "O2":16,  "PO4":17,
    "P4":18,  "P8":19,
    "CP6":20, "CP2":21,
    "C4":22,  "T8":23,
    "FC6":24, "FC2":25,
    "F4":26,  "F8":27,
    "AF4":28, "Fp2":29,
    "Fz":30,  "Cz":31
}

laplace_array = [
    [7, 3, 6, 31, 11],
    [31, 30, 7, 12, 22],
    [22, 26, 31, 18, 23],
    [8, 4, 9, 13, 21],
    [21, 25, 8, 17, 20],
    [11, 7, 10, 14, 12],
    [12, 31, 11, 15, 18],
    [18, 22, 12, 16, 19]
]

def loadData(full_fname, fs_Hz):
    # load data into numpy array
    data = []
    arry = []
    target = []
    fname_data = full_fname
    counter = 0
    
    with open(fname_data) as data_file:
        while True:

            line = data_file.readline()
            if line == "":
                break
            arr = line.split()
            # if arr[-1] == "7.0000000e+000":
            #     continue
            counter += 1

            for i in range(len(arr)-1):
                #one = laplace_array[i][0]
                # two = laplace_array[i][1]
                # three = laplace_array[i][2]
                # four = laplace_array[i][3]
                # five = laplace_array[i][4]
                temp = float(arr[i])
                
                #temp = 4*float(arr[one])-float(arr[two])-float(arr[three])-float(arr[four])-float(arr[five]) 
                arry.append(temp)
            
            data.append(arry)

            arry = []
            # if counter == 512:
            target.append(float(arr[-1])-2.0)
            # counter = 0
            
    # # print '*******',len(data)
    return data,target
    #data = np.transpose(data[0:512])

def butter_bandpass_filter(data, highcut, fs_Hz, passlh, order=5):
    nyq = 0.5 * fs_Hz
    high = highcut / nyq
    b, a = signal.butter(order, high, btype=passlh)
    y = signal.lfilter(b, a, data)
    return y

def fft_plot(y):
    # Number of samplepoints
    N = 512
    # sample spacing
    T = 1.0 / 512.0
    x = np.linspace(0.0, N*T, N)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    fig, ax = plt.subplots()
    plt.ylim([0,40])
    plt.xlim([0,100])
    plt.xlabel("Frequency")
    plt.ylabel("Intensity")
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()


def convertToFreqDomain(f_eeg_data_uV, fs_Hz, NFFT, overlap):
       
    # compute spectrogram
    #fig = plt.figure(figsize=(7.5, 9.25))  # make new figure, set size in inches
    #ax1 = plt.subplot(311)
    spec_PSDperHz, freqs, t_spec = mlab.specgram(np.squeeze(f_eeg_data_uV),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
                                   
    # convert the units of the spectral data
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)  # convert to "per bin"
    del spec_PSDperHz  # remove this variable so that I don't mistakenly use it
    
    return spec_PSDperBin, t_spec, freqs