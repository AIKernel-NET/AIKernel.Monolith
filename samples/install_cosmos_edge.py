"""Install the local Edge example against the verified import, preserving existing registrations."""
import argparse
import hashlib
import json
from pathlib import Path


def install(model_root: Path) -> Path:
    template = Path(__file__).with_name("cosmos-edge.json")
    registration = json.loads(template.read_bytes())
    config = registration["cosmos"]
    manifest = model_root / config["directory"] / "import-manifest.json"
    data = manifest.read_bytes()
    # The recorded importer wrote CRLF on Windows and LF on Linux.
    recorded_bytes = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if hashlib.sha256(recorded_bytes).hexdigest() != config["manifestSha256"]:
        raise ValueError("Import manifest differs from the verified Edge example")
    config["manifestSha256"] = hashlib.sha256(data).hexdigest()
    target = model_root / "cosmos-edge.json"
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(registration, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        if json.loads(target.read_bytes()) != registration:
            raise ValueError(f"Existing registration differs: {target}") from None
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    args = parser.parse_args()
    print(install(args.model_root))
