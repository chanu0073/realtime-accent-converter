from preprocessing.manifest_builder import (
    ManifestBuilder
)

builder = ManifestBuilder()

builder.build(
    audio_dir="data",
    output_manifest="data/manifests/train.jsonl"
)