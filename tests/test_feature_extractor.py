from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor


loader = AudioLoader()
extractor = FeatureExtractor()

waveform = loader.load("data/sample.wav")

features = extractor.extract_all(waveform)

print("Mel Shape:", features["mel"].shape)
print("F0 Shape:", features["f0"].shape)
print("Energy Shape:", features["energy"].shape)