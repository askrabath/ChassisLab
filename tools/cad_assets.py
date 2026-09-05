"""Convert supplied STEP assemblies into cached visual meshes and manifests.

Canonical STEP files remain unchanged.  Generated meshes are for rendering;
physics collision geometry stays explicit and low complexity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "cad_assets"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sources(claw: Path, field_zip: Path) -> dict[str, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    claw_dir = ASSETS / "clawbot"
    field_dir = ASSETS / "override_field"
    claw_dir.mkdir(exist_ok=True)
    field_dir.mkdir(exist_ok=True)
    claw_step = claw_dir / "canonical.step"
    field_step = field_dir / "canonical.step"
    if not claw_step.exists() or claw_step.stat().st_size != claw.stat().st_size:
        shutil.copy2(claw, claw_step)
    if not field_step.exists():
        with zipfile.ZipFile(field_zip) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith((".step", ".stp"))]
            if len(members) != 1:
                raise ValueError(f"expected one field STEP member, found {len(members)}")
            with archive.open(members[0]) as source, field_step.open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    manifest = {
        "format_version": 1,
        "sources": {
            "clawbot": {
                "provided_path": str(claw),
                "canonical_path": str(claw_step.relative_to(ROOT)),
                "bytes": claw_step.stat().st_size,
                "sha256": sha256(claw_step),
                "authority": "user-provided Clawbot CAD; provenance not independently verified",
            },
            "override_field": {
                "provided_path": str(field_zip),
                "canonical_path": str(field_step.relative_to(ROOT)),
                "bytes": field_step.stat().st_size,
                "sha256": sha256(field_step),
                "authority": "user-provided ZIP identified as official VEX Override field CAD",
            },
        },
        "representations": {
            "canonical": "STEP source, never loaded in the simulation loop",
            "visual": "tessellated and decimated mesh",
            "collision": "separate primitives or validated convex pieces; visual mesh is not collision geometry",
            "metadata": "dimensions, provenance, transforms and unknown values",
        },
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"clawbot": claw_step, "override_field": field_step}


def tessellate(name: str, step: Path, linear_tolerance_mm: float, target_faces: int) -> Path:
    import trimesh

    directory = ASSETS / name
    raw = directory / "visual_raw.stl"
    visual = directory / "visual.obj"
    metadata_path = directory / "metadata.json"
    if visual.exists() and metadata_path.exists():
        print(f"{name}: cached {visual}", flush=True)
        return visual
    started = time.monotonic()
    if not raw.exists():
        import cadquery as cq
        print(f"{name}: reading {step.stat().st_size / 1e6:.1f} MB STEP", flush=True)
        model = cq.importers.importStep(str(step))
        shape = model.val()
        box = shape.BoundingBox()
        source_min = [box.xmin, box.ymin, box.zmin]
        source_max = [box.xmax, box.ymax, box.zmax]
        source_size = [box.xlen, box.ylen, box.zlen]
        print(f"{name}: bounds mm {box.xlen:.1f} x {box.ylen:.1f} x {box.zlen:.1f}", flush=True)
        cq.exporters.export(
            model,
            str(raw),
            tolerance=linear_tolerance_mm,
            angularTolerance=0.25,
        )
    else:
        print(f"{name}: reusing cached raw tessellation", flush=True)
    loaded = trimesh.load_mesh(raw, process=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    else:
        mesh = loaded
    if 'source_min' not in locals():
        source_min = mesh.bounds[0].tolist()
        source_max = mesh.bounds[1].tolist()
        source_size = (mesh.bounds[1] - mesh.bounds[0]).tolist()
        print(f"{name}: cached bounds mm " + " x ".join(f"{x:.1f}" for x in source_size), flush=True)
    before = len(mesh.faces)
    if before > target_faces:
        print(f"{name}: simplifying {before:,} to {target_faces:,} faces", flush=True)
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    mesh.remove_unreferenced_vertices()
    mesh.export(visual)
    metadata = {
        "format_version": 1,
        "source_sha256": sha256(step),
        "source_units": "millimeter (inferred from plausible assembly bounds; verify before manufacturing use)",
        "source_bounds_mm": {
            "min": source_min,
            "max": source_max,
            "size": source_size,
        },
        "visual_mesh": str(visual.relative_to(ROOT)),
        "raw_faces": before,
        "visual_faces": len(mesh.faces),
        "visual_vertices": len(mesh.vertices),
        "tessellation_tolerance_mm": linear_tolerance_mm,
        "elapsed_s": time.monotonic() - started,
        "collision_model": None,
        "mass_kg": None,
        "center_of_mass": None,
        "connection_points": None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"{name}: wrote {visual} ({len(mesh.faces):,} faces)", flush=True)
    return visual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "clawbot", "field", "all"])
    parser.add_argument("--claw", type=Path, default=Path("/Users/joshuamichealscarano/Downloads/claw bot.step"))
    parser.add_argument("--field-zip", type=Path, default=Path("/Users/joshuamichealscarano/Downloads/v5rc-override-fieldcad.zip"))
    args = parser.parse_args()
    sources = canonical_sources(args.claw, args.field_zip)
    if args.command in ("clawbot", "all"):
        tessellate("clawbot", sources["clawbot"], 0.8, 90_000)
    if args.command in ("field", "all"):
        tessellate("override_field", sources["override_field"], 2.5, 350_000)
    print(ASSETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
