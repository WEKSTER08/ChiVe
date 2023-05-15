import librosa

y, sr = librosa.load('data/aud1.wav')
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0)



mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr)

print ("f0 are =",f0,"c0 are =", mfcc_val)