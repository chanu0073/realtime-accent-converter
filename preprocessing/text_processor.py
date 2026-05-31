import re


class TextProcessor:
    """
    Basic text normalization for TTS training.

    Current features:
    - Lowercase conversion
    - Remove extra spaces
    - Remove unsupported symbols

    Future:
    - Number normalization
    - Phoneme conversion
    - CMU dictionary lookup
    """

    def normalize(self, text: str) -> str:
        """
        Normalize text for training.

        Example:
            Input:
                "Hello,   World!!!"

            Output:
                "hello world"
        """

        text = text.lower()

        text = re.sub(
            r"[^a-zA-Z0-9\s']",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip()

        return text