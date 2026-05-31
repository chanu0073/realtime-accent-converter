from pathlib import Path
import json

from preprocessing.audio_loader import AudioLoader


class ManifestBuilder:
    """
    Builds dataset manifest files.

    Current fields:
    - audio_path
    - duration

    Future fields:
    - transcript
    - speaker_id
    - accent
    """

    def __init__(self, sample_rate: int = 16000):
        self.loader = AudioLoader(sample_rate)

    def build(
        self,
        audio_dir: str,
        output_manifest: str,
    ):
        """
        Scan directory for wav files and create manifest.

        Args:
            audio_dir:
                Root directory containing wav files

            output_manifest:
                Path to output .jsonl manifest
        """

        audio_dir = Path(audio_dir)

        wav_files = sorted(
            audio_dir.rglob("*.wav")
        )

        print(f"Found {len(wav_files)} wav files")

        with open(output_manifest, "w") as f:

            for wav_path in wav_files:

                waveform = self.loader.load(
                    str(wav_path)
                )

                duration = self.loader.get_duration(
                    waveform
                )

                record = {
                    "audio_path": str(wav_path),
                    "duration": round(duration, 3),
                }

                f.write(
                    json.dumps(record)
                    + "\n"
                )

        print(
            f"Manifest saved to {output_manifest}"
        )