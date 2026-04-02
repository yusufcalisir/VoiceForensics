import numpy as np
import scipy.io.wavfile as wavf
import os

os.makedirs('test_audio', exist_ok=True)

# Generate a 5-second sine wave
sample_rate = 16000
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio1 = 0.5 * np.sin(2 * np.pi * 440 * t)
wavf.write('test_audio/audio1.wav', sample_rate, (audio1 * 32767).astype(np.int16))

# Same sine wave but 4 seconds
t2 = np.linspace(0, 4.0, int(sample_rate * 4.0))
audio2 = 0.5 * np.sin(2 * np.pi * 440 * t2)
wavf.write('test_audio/audio2.wav', sample_rate, (audio2 * 32767).astype(np.int16))

# Generate short white noise (1.5 seconds) to test short audio and different speaker
noise = np.random.normal(0, 0.1, int(sample_rate * 1.5))
wavf.write('test_audio/audio3.wav', sample_rate, (noise * 32767).astype(np.int16))

print("Created test audio files in test_audio/")
