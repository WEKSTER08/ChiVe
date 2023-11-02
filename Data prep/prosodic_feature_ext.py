import librosa
import math
import numpy as np
y, sr = librosa.load('data/wav/ISLE_SESS0003_BLOCKD01_03_sprt1.wav',sr=16000)
y1, sr1 = librosa.load('data/aud1.wav')
f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),frame_length= 20, hop_length= 10 )
times = librosa.times_like(f0)

print("sr is = ",sr,"len of y = ", len(y),"sr1 is = ",sr1,"len of y1 = ", len(y1))

mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr,hop_length= 10)

print ("f0 are =",f0.shape,"c0 are =", mfcc_val.shape)

duration = librosa.get_duration(y=y, sr=sr)
#frames per millisecond
fpms = math.ceil(len(f0)/(duration*1000))

print("frames per millisecond", fpms)

# #for phrnn input 
# f_avg = []
# # smoothening factor
# sf = 0.9
# for i in range(len(f0)-1):
#     diff = 0
#     if i == 0:
#         print(i)
#         f_avg[i] = f0[i]
       
#     else: 
#         diff = f0[i] - f0[i-1]
#         diff = diff*sf
#         f_avg[i] = f0[i-1] + diff

if f0[0] == 'nan':
    print("Hi")
print(f0[0].dtype)
