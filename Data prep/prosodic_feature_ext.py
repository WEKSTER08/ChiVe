
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

## To equal the length of the f0 and c0 values by averaging
def average_reduce(input_list, target_length):
    original_length = len(input_list)
    dimension = len(input_list[0])  # Assuming all elements have the same dimensionality
    step = original_length // target_length

    reduced_list = [
        np.mean(input_list[i:i+step], axis=0) for i in range(0, original_length, step)
    ]

    return reduced_list[:target_length]

def average_reduce_ph(input_list, target_length):
    original_length = len(input_list)
    dimension = len(input_list[0])  # Assuming all elements have the same dimensionality
    step = original_length // target_length

    reduced_list = [
        np.mean(input_list[i]) for i in range(0, original_length, step)
    ]
    print(reduced_list[:10])

    return reduced_list[:target_length]
# Load the audio file
# file_path = 'path/to/your/audio/file.wav'
def extract_f_c(data_path,ph):
    y, sr = librosa.load(data_path,sr=16000,)

    # Compute the short-time Fourier transform (STFT)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # Extract the frequency values
    frequencies = librosa.fft_frequencies(sr=16000)

    mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr,hop_length= 10)
    mfcc_val = mfcc_val.T
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'),frame_length= 20, hop_length= 10 )
    
    if ph ==1 : 
        print("ph")
        reduced_mfcc = average_reduce_ph(mfcc_val,len(frequencies))
    else: 
        print("No ph")
        reduced_mfcc = average_reduce(mfcc_val,len(frequencies))
    frnn_inp = []
    for i,vals in enumerate(reduced_mfcc):
        temp = []
        frnn_inp.append([vals,frequencies[i]])
    print(reduced_mfcc[:5])
    frequencies = frequencies.reshape(-1,1)
    # frnn_inp = np.concatenate((reduced_mfcc,frequencies),axis =1)
    return frnn_inp

f_c_inp = extract_f_c('data/wav/ISLE_SESS0003_BLOCKD01_03_sprt1.wav',1)
print((f_c_inp[:10]))
