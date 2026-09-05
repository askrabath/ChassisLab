"""Build disposable low-detail VISUAL meshes. Never used for collision or mass."""
import json
from pathlib import Path
import trimesh

ROOT=Path(__file__).resolve().parents[1]

def build():
    records=[]
    for name in ['clawbot','override_field']:
        folder=ROOT/'data/cad_assets'/name
        tree=json.loads((folder/'assembly_tree.json').read_text())
        for key,definition in tree['definitions'].items():
            source=ROOT/definition['mesh'];target=source.with_name(source.stem+'_lod.obj')
            if target.exists():continue
            mesh=trimesh.load(source,force='mesh')
            original=len(mesh.faces)
            if original>1800:
                mesh=mesh.simplify_quadric_decimation(face_count=1800)
                # Some compound CAD parts cannot collapse to the budget. A
                # convex visual envelope is acceptable; physics stays original.
                if len(mesh.faces)>3600:mesh=mesh.convex_hull
            mesh.export(target)
            records.append(dict(asset=name,part=key,original_faces=original,visual_faces=len(mesh.faces)))
        print(name+': visual cache ready',flush=True)
    (ROOT/'data/cad_assets/visual_lod_manifest.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':build()
