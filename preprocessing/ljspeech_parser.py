import json
from pathlib import Path

import soundfile as sf

from preprocessing.text_processor import TextProcessor


class LJSpeechParser:

    def __init__(self):
        self.text_processor = TextProcessor()

    def build_manifest(
        self,
        dataset_root: str,
        output_manifest: str,
    ):
        dataset_root = Path(dataset_root)

        metadata_path = dataset_root / "metadata.csv"
        wav_dir = dataset_root / "wavs"

        total_entries = 0

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as metadata_file:

            with open(
                output_manifest,
                "w",
                encoding="utf-8"
            ) as output_file:

                for row_num, line in enumerate(
                    metadata_file,
                    start=1
                ):

                    row = line.rstrip(
                        "\n"
                    ).split(
                        "|",
                        maxsplit=2
                    )

                    if len(row) != 3:
                        print(
                            f"Skipping malformed line "
                            f"{row_num}"
                        )
                        continue

                    file_id = row[0]

                    raw_text = row[1]

                    normalized_text = (
                        self.text_processor
                        .normalize(row[2])
                    )

                    wav_path = (
                        wav_dir /
                        f"{file_id}.wav"
                    )

                    info = sf.info(
                        str(wav_path)
                    )

                    duration = (
                        info.frames /
                        info.samplerate
                    )

                    record = {
                        "id": file_id,
                        "audio_path": str(
                            wav_path
                        ),
                        "raw_transcript": raw_text,
                        "transcript": normalized_text,
                        "duration": round(
                            duration,
                            3
                        ),
                    }

                    output_file.write(
                        json.dumps(record)
                        + "\n"
                    )

                    total_entries += 1

        print(
            f"Created manifest with "
            f"{total_entries} entries"
        )