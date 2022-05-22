import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

# Relative dir. for dataset (input)
dataset_dir = "../data/data_preprocessed_matlab_full/"
# Relative dir. for output
result_dir = "../data/1D_dataset/"

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_data_from_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

#########################################################
# For each data file "s01.mat", preprocess the data
# Input   'file'    : full path to dataset file, it is for 1 person
# Output1 'base_DE' :
# Output2 'trial_DE':
############################################################
def decompose(file): 
    # pre-trail part need to be considered seperately
    frequency = 128 # the EEG signals were down-sampled to 128 Hz
    start_time = 3  # 3s pre-trial signals
    start_index = start_time* frequency   # 3*128=384

    # trial*channel*sample
    data = get_data_from_file(file)
    shape = data.shape# 40*40*8064
    #print(shape)
    breakpoint()

    decomposed_de = np.empty([0, 4, 60])
    base_DE = np.empty([0, 128])

    for trial in range(40): # there are 40 videos
    # for each video            
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0]) # init theta DE list
        temp_base_alpha_DE = np.empty([0]) # init alpha DE list
        temp_base_beta_DE = np.empty([0])  # init beta  DE list
        temp_base_gamma_DE = np.empty([0]) # init gamma DE list

        temp_de = np.empty([0, 60])
        # there are 40 channel
        for channel in range(32): # we only take first 32 channels
            trial_signal = data[trial, channel, start_index:] #from 384 to end
            base_signal = data[trial, channel,  :start_index] #from begin to 384
            # ****************compute base (3 sec) DE****************
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)   # apply bandpass filter
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            base_theta_DE = (compute_DE(base_theta[:128]) + compute_DE(base_theta[128:256]) + compute_DE( # calculate Differential entropy
                base_theta[256:])) / 3  #128Hz, 1s: [0:128], 2s:[128:256], 3s:[256:384]
            base_alpha_DE = (compute_DE(base_alpha[:128]) + compute_DE(base_alpha[128:256]) + compute_DE( # just keep average for 3 sec
                base_alpha[256:])) / 3
            base_beta_DE = (compute_DE(base_beta[:128]) + compute_DE(base_beta[128:256]) + compute_DE(
                base_beta[256:])) / 3
            base_gamma_DE = (compute_DE(base_gamma[:128]) + compute_DE(base_gamma[128:256]) + compute_DE(
                base_gamma[256:])) / 3

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE) # append theta DE list with base part
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE) # append alpha DE list
            temp_base_beta_DE  = np.append(temp_base_beta_DE, base_beta_DE)   # append beta  DE list
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE) # append gamma DE list

            # ****************compute trial (60 sec) DE****************
            # theta (3 - 7 Hz), alpha (8 - 13 Hz), beta (14 - 29 Hz), and gamma (30 - 47 Hz） from DEAP dataset paper
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)  # apply bandpass filter
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(60): #trial part is 60s for 1 video
                DE_theta = np.append(DE_theta, compute_DE(theta[index * frequency:(index + 1) * frequency]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * frequency:(index + 1) * frequency]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * frequency:(index + 1) * frequency]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * frequency:(index + 1) * frequency]))
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha]) # connect theta_DE and alpha_DE
            temp_de = np.vstack([temp_de, DE_beta])  # then connect beta_DE
            temp_de = np.vstack([temp_de, DE_gamma] )# then connect gamma_DE
            #end for each second (in total 60)  
        #end for each channel (in total 32)  
        # here we have concatenated temp_de (for trial DE)
        temp_trial_de = temp_de.reshape(-1, 4, 60) # ?* 4 frequencies * 60 seconds
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)  # connect theta_DE and alpha_DE
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)   # then connect beta_DE
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)   # then connect gamma_DE
        base_DE = np.vstack([base_DE, temp_base_DE])
    #end for each video  (in total 40)  
    decomposed_de = decomposed_de.reshape(-1, 32, 4, 60).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)
                        # ?(40 videos)* 32 EEG channels * 4 frequencies * 60 seconds
                        # change axis order
                        # ? * 4 frequencies * 32 EEG channels
                        # ？(40*60) * 128 （4*32=128）

    print("base_DE shape:", base_DE.shape)         # 40*128    (videos)*(frequencies*channels)
    print("trial_DE shape:", decomposed_de.shape)  # 2400*128  (videos*seconds)*(frequencies*channels)
    #breakpoint()
    return base_DE, decomposed_de


def get_labels(file):
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    #shape = sio.loadmat(file)["labels"].shape# 40 videos *4 labels
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels 
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    dominance_labels = sio.loadmat(file)['labels'][:, 2] > 5  # dominance labels
    # liking labels are ignored
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels = np.empty([0])
    for i in range(len(valence_labels)):
    #for each video (in total 40)
        for j in range(0, 60):
        #for each second (in total 60)
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
            final_dominance_labels = np.append(final_dominance_labels, dominance_labels[i])
    print("labels:", final_arousal_labels.shape) #2400
    #breakpoint()
    return final_arousal_labels, final_valence_labels, final_dominance_labels

'''
unused
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
'''

'''
unused
def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized
'''

if __name__ == '__main__':

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir): # for all files under "data_preprocessed_matlab" folder
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        base_DE, trial_DE = decompose(file_path)                                 # preprocess the data
        arousal_labels, valence_labels, dominance_labels = get_labels(file_path) #get labels
        sio.savemat(result_dir + "DE_" + file, # path and file name
                    {"base_data": base_DE,    #input 1                   # 40*128    (videos)*(frequencies*channels)
                     "data": trial_DE,    #input 2                       # 2400*128  (videos*seconds)*(frequencies*channels)
                     "valence_labels": valence_labels,#label 1           # 2400      (videos*seconds)
                     "arousal_labels": arousal_labels,#label 2           # 2400      (videos*seconds)
                     "dominance_labels": dominance_labels})#label 3      # 2400      (videos*seconds)
        #breakpoint()
