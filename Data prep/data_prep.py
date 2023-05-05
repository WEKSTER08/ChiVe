import deepspeech
import numpy as np
import pyaudio

# Initialize DeepSpeech model
model_path = "path/to/deepspeech/model.pb"
model = deepspeech.Model(model_path)

# Set up PyAudio to get audio input from microphone
chunk_size = 1024
sample_rate = 16000
pa = pyaudio.PyAudio()

# Start audio stream from microphone
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

print("Speak now!")

# Keep listening for audio input until user stops speaking
frames = []
while True:
    # Read audio data from microphone
    data = stream.read(chunk_size)
    frames.append(data)

    # Check if user has stopped speaking
    if np.mean(np.frombuffer(data, dtype=np.int16)) < 500:
        break

# Convert audio data to text using DeepSpeech
audio = np.concatenate(frames)
text = model.stt(audio)

print(f"You said: {text}")

# Use NLTK to extract linguistic features
from nltk import word_tokenize, pos_tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
print("Parts of speech:", pos_tags)

# Close audio stream and PyAudio instance
stream.stop_stream()
stream.close()
pa.terminate()
