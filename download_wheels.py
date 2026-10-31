import hashlib
import json
import pathlib
import time
import urllib.request

specs = [
    ("numpy", "1.23.5", "cp38-cp38-win_amd64.whl"),
    ("opencv-python", "4.8.0.76", "cp37-abi3-win_amd64.whl"),
]
out = pathlib.Path(r"G:\ddddd\9DTact-main\work-wheels")
out.mkdir(exist_ok=True)
chunk_size = 4 * 1024 * 1024
for name, version, suffix in specs:
    info = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30))
    file = next(item for item in info["urls"] if item["filename"].endswith(suffix))
    dest = out / file["filename"]
    temp = dest.with_suffix(dest.suffix + ".part")
    size = file["size"]
    print(f"Downloading {file['filename']} ({size} bytes)", flush=True)
    with temp.open("wb") as stream:
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size - 1, size - 1)
            for attempt in range(4):
                try:
                    req = urllib.request.Request(file["url"], headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=45) as response:
                        payload = response.read()
                    if len(payload) != end - start + 1:
                        raise IOError(f"Unexpected range length: {len(payload)}")
                    stream.write(payload)
                    print(f"  {end + 1}/{size}", flush=True)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2)
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    if digest != file["digests"]["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {dest.name}: {digest}")
    temp.replace(dest)
    print(f"Verified {dest.name} SHA256 {digest}", flush=True)
