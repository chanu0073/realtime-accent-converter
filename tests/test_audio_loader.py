from preprocessing.audio_loader import AudioLoader

loader = AudioLoader()

audio = loader.load("data/sample.wav")

print("Samples:", len(audio))
print("Duration:", loader.get_duration(audio))
print("Dtype:", audio.dtype)