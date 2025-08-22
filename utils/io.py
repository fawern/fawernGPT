import json
import os

def ensure_dir(
    path: str
):
    os.makedirs(path, exist_ok=True)

def write_json(
    path: str,
    obj: dict
):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(
    path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_text(
    path: str,
    text: str
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_text(
    path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
