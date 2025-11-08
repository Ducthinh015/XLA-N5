import os
import sys
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/infer/image")
IMAGE_PATH = os.environ.get("IMAGE_PATH")

if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print("Please set IMAGE_PATH env var to a valid image file, e.g.:")
    print("  $env:IMAGE_PATH=\"C:/path/to/sample.jpg\"")
    sys.exit(1)

with open(IMAGE_PATH, "rb") as f:
    files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
    r = requests.post(API_URL, files=files, timeout=60)

print("Status:", r.status_code)
try:
    print(r.json())
except Exception:
    print(r.text)
