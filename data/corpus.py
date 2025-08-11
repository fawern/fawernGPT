import os
from typing import Generator


def iter_text(
    path: str,
    lower_case: bool = True
) -> Generator[str, None, None]:
    """
    Iterate over the text in a file or directory.

    Args:
        path: Path to the file or directory to iterate over.
        lower_case: Whether to lowercase the text.

    Returns:
        A generator of strings, one for each line in the file or directory.
    """
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                fp = os.path.join(root, file)

                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip("\n\r")
                        yield line.lower() if lower_case else line
    
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n\r")
                yield line.lower() if lower_case else line