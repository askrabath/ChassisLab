"""Build a conservative VEX component index from supplied STEP products.

This is the local-file ingestion layer.  Onshape discovery/download can feed
the same manifest later, but this tool never labels an assembly inventory as a
complete VEX parts library.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vex_parts"
PRODUCT = re.compile(r"\bPRODUCT\s*\(\s*'((?:''|[^'])*)'", re.DOTALL)
PART_NUMBER = re.compile(r"\b(?:27[56])-\d{4}(?:-\d{3})?\b")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def category(name: str) -> str:
    value = name.lower()
    rules = [
        ("motor", "motors"), ("battery", "electronics"), ("brain", "electronics"),
        ("wheel", "wheels"), ("gear", "gears"), ("sprocket", "sprockets"),
        ("shaft", "shafts"), ("bearing", "bearings"), ("collar", "collars"),
        ("spacer", "spacers"), ("standoff", "fasteners"), ("screw", "fasteners"),
        ("nut", "fasteners"), ("retainer", "fasteners"), ("gusset", "structure"),
        ("channel", "structure"), ("angle", "structure"), ("claw", "mechanisms"),
    ]
    return next((result for token, result in rules if token in value), "uncategorized")


def products(step: Path) -> list[str]:
    # STEP records may span lines. Product text is small relative to geometry,
    # so a single read keeps parsing deterministic and simple for these files.
    text = step.read_text(errors="replace")
    def decode(value):
        value=re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\",lambda m:bytes.fromhex(m[1]).decode('utf-16-be'),value)
        return value.replace("''", "'").replace("\n", " ").replace('\u200e','').strip()
    return [decode(m.group(1)) for m in PRODUCT.finditer(text)]


def discover(step: Path, source_name: str) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    counts = Counter(products(step))
    records = []
    for name, occurrences in sorted(counts.items()):
        if not name:
            continue
        number = PART_NUMBER.search(name)
        identity = (number.group(0) if number else re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))
        records.append({
            "id": identity,
            "name": name,
            "part_number": number.group(0) if number else None,
            "category": category(name),
            "product_definition_occurrences": occurrences,
            "canonical_geometry": None,
            "simulation_mesh": None,
            "configuration": None,
            "material": None,
            "mass_kg": None,
            "bounding_box_m": None,
            "center_of_mass_m": None,
            "connection_points": None,
            "legal_vrc": None,
            "source": {
                "kind": "supplied_step_assembly",
                "assembly": source_name,
                "path": str(step.relative_to(ROOT)),
            },
        })
    manifest = {
        "format_version": 1,
        "scope": "components referenced by supplied assemblies; not the complete VEX V5 library",
        "source_sha256": digest(step),
        "component_definitions": len(records),
        "parts": records,
    }
    output = DATA / f"{source_name}_manifest.json"
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), "definitions": len(records), "categories": Counter(x["category"] for x in records)}, default=dict, indent=2))
    return manifest


def build_index() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads(path.read_text()) for path in sorted(DATA.glob("*_manifest.json"))]
    index = {}
    collisions = []
    for manifest in manifests:
        for part in manifest["parts"]:
            key = part["id"]
            if key in index and index[key]["name"] != part["name"]:
                collisions.append({"id": key, "names": [index[key]["name"], part["name"]]})
                key = f"{key}_{len(collisions)}"
            index[key] = part
    result = {"format_version": 1, "parts": index, "identity_collisions": collisions}
    (DATA / "index.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"indexed": len(index), "identity_collisions": len(collisions)}, indent=2))
    return result


def validate() -> dict:
    index = json.loads((DATA / "index.json").read_text())
    problems = []
    for key, part in index["parts"].items():
        if not key or not part["name"] or not part["category"]:
            problems.append({"id": key, "problem": "missing required identity metadata"})
        if part["mass_kg"] is not None and not 0 < part["mass_kg"] < 20:
            problems.append({"id": key, "problem": "implausible mass"})
    report = {"valid": not problems, "indexed": len(index["parts"]), "problems": problems,
              "unknown_fields_preserved_as_null": True}
    (DATA / "import_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("step", type=Path)
    discover_parser.add_argument("--name", required=True)
    sub.add_parser("build-index")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "discover":
        discover(args.step.resolve(), args.name)
    elif args.command == "build-index":
        build_index()
    else:
        validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
