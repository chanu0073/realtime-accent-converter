from preprocessing.text_processor import TextProcessor

processor = TextProcessor()

examples = [
    "Hello World!!!",
    "How are     you?",
    "I LOVE Speech AI.",
]

for text in examples:
    print("Input :", text)
    print("Output:", processor.normalize(text))
    print("-" * 30)