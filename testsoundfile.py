import soundfile as sf
data, sr = sf.read("C:/Users/gigggjjffd/Desktop/work/machine learning/VAE/LJSpeech-1.1/LJSpeech-1.1/wavs/LJ001-0001.wav")
print(data.shape, sr)