
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
    y, sr = librosa.load(data_path,sr=16000)

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

# f_c_inp = extract_f_c('data/wav/ISLE_SESS0003_BLOCKD01_03_sprt1.wav',0)
# print((f_c_inp[:10]))



data_path = 'data/wav/ISLE_SESS0003_BLOCKD01_03_sprt1.wav'
y, sr = librosa.load(data_path,sr=16000)
mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr,hop_length= 10)
print(len(mfcc_val.T[:4096]))
f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'),frame_length= 40, hop_length= 20 )
# f0, voiced_flag = librosa.effects.harmonic(y)
# print(len(f0),f0[:600])
## Test f0 values

## Test to find f0 values


# import aubio
# import numpy as np

# # Load audio file
# filename = 'data/aud1.wav'
# y, sr = librosa.load(filename,sr=16000)

# # Set parameters
# hop_size = 10
# frame_size = 20

# # Create pitch object
# pitch_o = aubio.pitch("yin", frame_size, hop_size, sr)

# # Initialize array to store pitch values
# pitch_values = []

# # Process audio in frames
# for i in range(0, len(y), hop_size):
#     frame = y[i:i+frame_size]

#     # Get pitch value for the frame
#     pitch = pitch_o(frame)[0]
#     pitch_values.append(pitch)


# print(pitch_values)

from pydub import AudioSegment
from pydub.playback import play
# # from swipe import swipe

# Load audio file
audio = AudioSegment.from_file(data_path, format="wav")

# Extract raw audio data
y = audio.raw_data

# # Set sample rate
# sample_rate = audio.frame_rate

# # Perform pitch estimation using SWIPE
# # pitch, _ = swipe(y, fs=22000)
# # print(pitch)
# # 'pitch' contains the estimated pitch values
# from sptk import *
# f0 = pysptk.sptk.swipe(y, 22000, 40, min=60.0, max=240.0, threshold=0.3, otype='f0')
# print(len(f0))

import pysptk
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load audio file
audio_file = "your_audio_file.wav"
sampling_rate, x = wavfile.read(data_path)

# Pre-emphasis (optional but often used)
# x = pysptk.process.preemphasis(x, coef=0.97)
print(sampling_rate)
# Get pitch (f0) using SWIPE algorithm
f0 = pysptk.swipe(x.astype(float), fs=sampling_rate/2, hopsize=10, min=60, max=240)

# Get cepstral (c0) coefficients using mcep
order = 24  # Order of cepstral coefficients
frame_length = 1024
hop_length = 80

# Use mcep to calculate cepstral coefficients
# c0 = pysptk.mcep(x, order,etype=1,eps=0.1)
print(mfcc_val.T[0])
print("Swipe output len -",len(f0))
# Visualize pitch (f0)
plt.plot(f0)
plt.title("Pitch (f0)")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.savefig("pitch.png")

# order = 25

# # Frame the audio signal (you may need to adjust the frame length and hop size)
# frames = librosa.util.frame(x.astype(float), frame_length=20, hop_length=10)

# # Apply a window function (e.g., Blackman window)
# # frames *= pysptk.blackman(n=20)

# # Get mel-cepstral coefficients
# cepstral_coefficients = pysptk.sptk.mcep(frames, order=order, alpha=0.42)

# # Display the first few coefficients for the first frame
# print("Cepstral Coefficients:")
# print(cepstral_coefficients[0, :])