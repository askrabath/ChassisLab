"""Build a MuJoCo inspection scene from a cached CAD visual mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def build_scene(name: str) -> tuple[Path, dict]:
    directory = ROOT / "data" / "cad_assets" / name
    metadata = json.loads((directory / "metadata.json").read_text())
    bounds = metadata["source_bounds_mm"]
    lower, upper = bounds["min"], bounds["max"]
    center = [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2]
    # STEP coordinates remain untouched in the cached mesh. This body transform
    # centers X/Y and puts the source's minimum Z on the MuJoCo floor.
    position = [-center[0] / 1000, -center[1] / 1000, -lower[2] / 1000]
    root = ET.Element("mujoco", model=f"cad_inspection_{name}")
    ET.SubElement(root, "compiler", meshdir=str(directory), autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81", timestep=".002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="800")
    ET.SubElement(visual, "headlight", ambient=".55 .55 .55", diffuse=".8 .8 .8")
    asset = ET.SubElement(root, "asset")
    visual_file = Path(metadata["visual_mesh"]).name
    ET.SubElement(asset, "mesh", name="cad_visual", file=visual_file, scale=".001 .001 .001")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 .1", rgba=".18 .21 .24 1")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1")
    body = ET.SubElement(world, "body", name="cad_assembly", pos=" ".join(map(str, position)))
    ET.SubElement(
        body,
        "geom",
        name="cad_visual_only",
        type="mesh",
        mesh="cad_visual",
        contype="0",
        conaffinity="0",
        group="1",
        rgba=".72 .75 .78 1",
    )
    # A visual wireframe envelope documents scale but is excluded from contact.
    size = [x / 2000 for x in bounds["size"]]
    envelope_pos = [0, 0, size[2]]
    ET.SubElement(
        world,
        "geom",
        name="source_bounds",
        type="box",
        pos=" ".join(map(str, envelope_pos)),
        size=" ".join(map(str, size)),
        contype="0",
        conaffinity="0",
        group="2",
        rgba=".1 .65 .9 .06",
    )
    scene = directory / "inspection.xml"
    scene.write_text(ET.tostring(root, encoding="unicode"))
    return scene, metadata


def render(name: str) -> Path:
    scene, metadata = build_scene(name)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dimensions = [x / 1000 for x in metadata["source_bounds_mm"]["size"]]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, dimensions[2] * 0.42]
    camera.distance = max(dimensions) * 1.65
    camera.azimuth = 130
    camera.elevation = -28 if name == "clawbot" else -62
    with mujoco.Renderer(model, height=800, width=1200) as renderer:
        renderer.update_scene(data, camera=camera)
        output = scene.with_name("inspection.png")
        Image.fromarray(renderer.render()).save(output)
    summary = {
        "scene": str(scene.relative_to(ROOT)),
        "screenshot": str(output.relative_to(ROOT)),
        "finite": all(map(lambda x: x == x, data.qpos)),
        "visual_only": True,
        "physical_collision_geometry": False,
        "source_bounds_m": dimensions,
    }
    scene.with_name("inspection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["clawbot", "override_field"])
    args = parser.parse_args()
    render(args.name)
