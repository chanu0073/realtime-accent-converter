from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor
from utils.visualizer import Visualizer


loader = AudioLoader()
extractor = FeatureExtractor()

waveform = loader.load("data/sample.wav")

features = extractor.extract_all(waveform)

Visualizer.plot_mel(features["mel"])
Visualizer.plot_f0(features["f0"])