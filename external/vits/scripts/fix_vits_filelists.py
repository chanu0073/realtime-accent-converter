from pathlib import Path

DATASET_PATH = (
    "/home/chanu/Files/projects/AIR/realtime-accent-converter/datasets/LJSpeech-1.1/wavs"
)

filelists_dir = Path(
    "filelists"
)

print("Exists:", filelists_dir.exists())
print("Absolute:", filelists_dir.resolve())

files = list(filelists_dir.glob("ljs*.cleaned"))

print("Found files:", len(files))

for file in files:
    print("Processing:", file)

    text = file.read_text(
        encoding="utf-8"
    )

    text = text.replace(
        "DUMMY1",
        DATASET_PATH
    )

    file.write_text(
        text,
        encoding="utf-8"
    )

    print(
        f"Updated: {file.name}"
    )