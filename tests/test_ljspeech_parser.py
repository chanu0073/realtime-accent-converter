from preprocessing.ljspeech_parser import (
    LJSpeechParser
)

parser = LJSpeechParser()

parser.build_manifest(
    dataset_root="datasets/LJSpeech-1.1",
    output_manifest="data/manifests/ljspeech.jsonl"
)