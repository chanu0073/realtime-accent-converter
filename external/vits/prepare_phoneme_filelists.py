from phonemizer.backend import EspeakBackend


backend = EspeakBackend(
    language="en-us",
    preserve_punctuation=False,
    with_stress=True
)


def process_file(input_path, output_path):

    lines_out = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)

    print("Processing:", total)

    for idx, line in enumerate(lines):

        wav_path, text = line.strip().split("|", 1)

        phonemes = backend.phonemize(
            [text],
            strip=True
        )[0]

        lines_out.append(
            f"{wav_path}|{phonemes}\n"
        )

        if idx % 1000 == 0:
            print(idx, "/", total)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines_out)

    print("Saved:", output_path)


process_file(
    "filelists/native_train.txt",
    "filelists/native_train_phonemes.txt"
)

process_file(
    "filelists/native_val.txt",
    "filelists/native_val_phonemes.txt"
)

print("DONE")