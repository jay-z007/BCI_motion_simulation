import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack

data_to_be_referred = [7,8,11,12,18,21,22,31]
electrode_map = {
    "Fp1":0,
    "AF3":1,
    "F7":2,
    "F3":3,
    "FC1":4,
    "FC5":5,
    "T7":6,
    "C3":7,
    "CP1":8,
    "CP5":9,
    "P7":10,
    "P3":11,
    "Pz":12,
    "PO3":13,
    "O1":14,
    "Oz":15,
    "O2":16,
    "PO4":17,
    "P4":18,
    "P8":19,
    "CP6":20,
    "CP2":21,
    "C4":22,
    "T8":23,
    "FC6":24,
    "FC2":25,
    "F4":26,
    "F8":27,
    "AF4":28,
    "Fp2":29,
    "Fz":30,
    "Cz":31
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

# laplace_array = [
#     ["C3", "F3", "T7", "Cz", "P3"],
#     ["Cz", "Fz", "C3", "Pz", "C4"],
#     ["C4", "F4", "Cz", "P4", "T8"],
#     ["CP1", "FC1", "CP5", "PO3", "CP2"],
#     ["CP2", "FC2", "CP1", "PO4", "CP6"],
#     ["P3", "C3", "P7", "O1", "Pz"],
#     ["Pz", "Cz", "P3", "Oz", "P4"],
#     ["P4", "C4", "Pz", "O2", "P8"]
# ]
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
            if arr[-1] == "7.0000000e+000":
                continue
            counter += 1

            for i in range(len(laplace_array)):
                one = laplace_array[i][0]
                two = laplace_array[i][1]
                three = laplace_array[i][2]
                four = laplace_array[i][3]
                five = laplace_array[i][4]
                
                temp = 4*float(arr[one])-float(arr[two])-float(arr[three])-float(arr[four])-float(arr[five]) 
                arry.append(temp)
            
            data.append(arry)
            # print counter
            arry = []
            if counter == 512:
                target.append(float(arr[-1])-2.0)
                counter = 0
    # print '*******',len(data)
    return data,target
    #data = np.transpose(data[0:512])

       # parse the data
       # data_indices = data[:, [0]]  # the first column is the packet index
    #eeg_data_uV = data[index]       # the 3rd column is EEG channel 2
   

def butter_bandpass_filter(data, highcut, fs_Hz, passlh, order=5):
    # hp_cutoff_Hz = 1.0
    # b, a = signal.butter(2, hp_cutoff_Hz/(fs_Hz / 2.0), 'highpass')  # define the filter
    # f_eeg_data_uV = signal.lfilter(b, a, data, 0) # apply along the zeroeth dimension

   # notch filter the data to remove 60 Hz and 120 Hz
    # notch_freq_Hz = np.array([50.0])  # these are the center frequencies
    # for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each center freq
    #     bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
    #     b, a = signal.butter(3, bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')  # create the filter
    #     f_eeg_data_uV = signal.lfilter(b, a, f_eeg_data_uV, 0)  # apply along the zeroeth dimension
    

    nyq = 0.5 * fs_Hz
    # low = lowcut / nyq
    high = highcut / nyq
    # b, a = signal.butter(order, 30/nyq, btype='low')
    # data = signal.lfilter(b, a, data)

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
    
    # make sine wave to test the scaling
    #f_eeg_data_uV = np.sin(2.0*np.pi*(fs_Hz / (NFFT-1))*10.0*1.0*t_sec)
    #f_eeg_data_uV = np.sqrt(2)*f_eeg_data_uV * np.sqrt(2.0)
    
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

def assessAlphaAndGuard(full_t_spec, freqs, full_spec_PSDperBin, alpha_band_Hz, guard_band_Hz):
    # compute alpha vs time
    bool_inds = (freqs > alpha_band_Hz[0]) & (freqs < alpha_band_Hz[1])
    alpha_max_uVperSqrtBin = np.sqrt(np.amax(full_spec_PSDperBin[bool_inds, :], 0))
    # alpha_sum_uVrms = np.sqrt(np.sum(full_spec_PSDperBin[bool_inds, :],0))
    
    bool_inds = ((freqs > guard_band_Hz[0][0]) & (freqs < guard_band_Hz[0][1]) |
                 (freqs > guard_band_Hz[1][0]) & (freqs < guard_band_Hz[1][1]))
    guard_mean_uVperSqrtBin = np.sqrt(np.mean(full_spec_PSDperBin[bool_inds, :], 0))
    alpha_guard_ratio = alpha_max_uVperSqrtBin / guard_mean_uVperSqrtBin
    
    return alpha_max_uVperSqrtBin, guard_mean_uVperSqrtBin, alpha_guard_ratio

def findTrueAndFalseDetections(full_t_spec,
                               alpha_max_uVperSqrtBin,
                               guard_mean_uVperSqrtBin,
                               alpha_guard_ratio,
                               t_lim_sec,
                               alpha_lim_sec,
                               detection_rule_set,
                               thresh1,
                               thresh2):
                               
    bool_inTime = (full_t_spec >= t_lim_sec[0]) & (full_t_spec <= t_lim_sec[1])
    bool_inTrueTime = np.zeros(full_t_spec.shape,dtype='bool')
    for lim_sec in alpha_lim_sec:
        bool_inTrueTime = bool_inTrueTime | ((full_t_spec >= lim_sec[0]) & (full_t_spec <= lim_sec[1]))    
    bool_inTrueTime =bool_inTrueTime[bool_inTime] # only keep those points within t_lim_sec
 
    #all three rule sets test the alpha amplitude
    bool_alpha_thresh = (alpha_max_uVperSqrtBin > thresh1)
    
    #the second test changes depending upon the rule set
    if (detection_rule_set == 1):
        bool_detect = bool_alpha_thresh[bool_inTime]
    elif (detection_rule_set == 2):
        bool_detect = bool_alpha_thresh[bool_inTime] & (guard_mean_uVperSqrtBin[bool_inTime] < thresh2)
    elif (detection_rule_set == 3):
        bool_detect = bool_alpha_thresh[bool_inTime] & (alpha_guard_ratio[bool_inTime] > thresh2)
    elif (detection_rule_set == 4):
        bool_alpha_thresh[2:-1] = bool_alpha_thresh[1:-2] | bool_alpha_thresh[2:-1]  #copy "true" to next time as well
        bool_guard = guard_mean_uVperSqrtBin < thresh2
        bool_guard[2:-1] = bool_guard[1:-2] & bool_guard[2:-1]  #copy "false" to next time as well
        bool_detect = bool_alpha_thresh[bool_inTime] & bool_guard[bool_inTime]
            
        
    #count true or false detections
    bool_true = bool_detect & bool_inTrueTime
    N_true = np.count_nonzero(bool_true)
    N_eyesClosed = np.count_nonzero(bool_inTrueTime)  #number of potential True detections
    bool_false = bool_detect & ~bool_inTrueTime
    N_false = np.count_nonzero(bool_false)
    N_eyesOpen = np.count_nonzero(~bool_inTrueTime)
    
    
    return N_true, N_false, N_eyesClosed, N_eyesOpen, bool_true, bool_false, bool_inTrueTime

def computeROC(N_true, N_false, N_eyesClosed, thresh1, thresh2, plot_N_false):
    shape = N_true.shape
    if (len(shape)==3):
        n_col_out = shape[3-1]
        use_third_dim = 1
    else:
        n_col_out = 1
        use_third_dim = 0
            
    plot_best_N_true = np.zeros([plot_N_false.size,n_col_out])
    plot_best_N_true_frac = np.zeros(plot_best_N_true.shape)
    plot_best_thresh1 = np.zeros(plot_best_N_true.shape)
    plot_best_thresh2 = np.zeros(plot_best_N_true.shape)
    for Icol in range(n_col_out):
        if use_third_dim:
            N_true_foo = N_true[:, :, Icol] 
            N_false_foo = N_false[:, :, Icol]
        else:
            N_true_foo = N_true 
            N_false_foo = N_false           
        
        for I_N_false in range(plot_N_false.size):
            bool = (N_false_foo == plot_N_false[I_N_false]);
            if np.any(bool):
                
                plot_best_N_true[I_N_false, Icol] = np.max(N_true_foo[bool])
                
                foo = np.copy(N_true_foo)
                foo[~bool] = 0.0  # some small value to all values at a different N_false
                inds = np.unravel_index(np.argmax(foo), foo.shape)
                plot_best_thresh1[I_N_false, Icol] = thresh1[inds[0]]
                plot_best_thresh2[I_N_false, Icol] = thresh2[inds[1]]
                
            # never be smaller than the previous value
            if (I_N_false > 0):
                if (plot_best_N_true[I_N_false-1,Icol] > plot_best_N_true[I_N_false,Icol]):
                    plot_best_N_true[I_N_false,Icol] = plot_best_N_true[I_N_false-1,Icol]
                    plot_best_thresh1[I_N_false, Icol] = plot_best_thresh1[I_N_false-1, Icol]
                    plot_best_thresh2[I_N_false, Icol] = plot_best_thresh2[I_N_false-1, Icol]
        
        if (use_third_dim):
            N_total = N_eyesClosed[Icol]
        else:
            N_total = N_eyesClosed
        plot_best_N_true_frac[:, Icol] = (plot_best_N_true[:, Icol]) / N_total
        
    return plot_best_N_true, plot_best_N_true_frac, plot_best_thresh1, plot_best_thresh2
