from datasets.tts_dataset import TTSDataset

dataset = TTSDataset(
    "data/manifests/ljspeech.jsonl"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("ID:", sample["id"])
print("Text:", sample["transcript"][:50])

print("Waveform:", sample["waveform"].shape)
print("Mel:", sample["mel"].shape)
print("F0:", sample["f0"].shape)
print("Energy:", sample["energy"].shape)