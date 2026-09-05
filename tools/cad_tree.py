"""STEP/XCAF assembly importer with instanced visual meshes and per-part cache.

No exact CAD bounding-box optimization or whole-assembly mesh simplification.
Mesh bounds are computed after tessellation. Unknown engineering data stays null.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def import_tree(name):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence,TDF_Label
    from OCP.TDataStd import TDataStd_Name
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE,TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorType
    import trimesh

    out=ROOT/'data'/'cad_assets'/name/'parts';out.mkdir(parents=True,exist_ok=True)
    source=out.parent/'canonical.step'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    signature=out.parent/'cache_signature.json'
    expected=dict(source_sha256=source_hash,linear_deflection_mm=.6,angular_deflection_rad=.28)
    if signature.exists() and json.loads(signature.read_text())!=expected:
        raise ValueError('CAD source or tessellation settings changed. Use a fresh asset cache directory to avoid stale meshes.')
    completed=out.parent/'assembly_tree.json'
    if completed.exists():
        cached=json.loads(completed.read_text())
        if cached['source_sha256']!=source_hash:
            raise ValueError('Source hash differs from assembly cache; use a fresh asset directory.')
        if all((ROOT/d['mesh']).exists() for d in cached['definitions'].values()):
            signature.write_text(json.dumps(expected,indent=2))
            print(f"{name}: using verified source cache ({len(cached['instances'])} instances)",flush=True)
            return cached
    signature.write_text(json.dumps(expected,indent=2))
    started=time.monotonic()
    doc=TDocStd_Document(TCollection_ExtendedString('cad'))
    reader=STEPCAFControl_Reader();reader.SetColorMode(True);reader.SetNameMode(True)
    print(name+': reading STEP records',flush=True)
    if int(reader.ReadFile(str(source)))!=1:raise RuntimeError('STEP reader failed')
    print(name+': transferring assembly tree',flush=True)
    if not reader.Transfer(doc):raise RuntimeError('STEP transfer failed')
    shape_tool=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors=XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    roots=TDF_LabelSequence();shape_tool.GetFreeShapes(roots)
    print(name+f': {roots.Length()} root assemblies transferred in {time.monotonic()-started:.1f}s',flush=True)
    definitions={};instances=[]

    def label_name(label):
        attr=TDataStd_Name()
        return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),attr) else 'unnamed'

    def matrix(location):
        transform=location.Transformation();a=np.eye(4)
        for i in range(3):
            for j in range(4):a[i,j]=transform.Value(i+1,j+1)
        return a

    def walk(label,parent_matrix,path):
        location=shape_tool.GetLocation_s(label)
        world=parent_matrix@matrix(location)
        reference=TDF_Label()
        if shape_tool.GetReferredShape_s(label,reference):definition=reference
        else:definition=label
        title=label_name(definition);path=path+[title]
        children=TDF_LabelSequence()
        if shape_tool.GetComponents_s(definition,children):
            for i in range(1,children.Length()+1):walk(children.Value(i),world,path)
            return
        key=f'p{definition.Tag():05d}'
        if key not in definitions:
            shape=shape_tool.GetShape_s(definition).Located(TopLoc_Location())
            if shape.IsNull():return
            meshfile=out/(key+'.obj');metafile=out/(key+'.json')
            if meshfile.exists() and metafile.exists():meta=json.loads(metafile.read_text())
            else:
                mesher=BRepMesh_IncrementalMesh(shape,.6,False,.28,True);mesher.Perform()
                vertices=[];faces=[];explorer=TopExp_Explorer(shape,TopAbs_FACE)
                while explorer.More():
                    face=TopoDS.Face_s(explorer.Current());loc=TopLoc_Location()
                    tri=BRep_Tool.Triangulation_s(face,loc)
                    if tri is not None:
                        start=len(vertices);trsf=loc.Transformation()
                        for n in range(1,tri.NbNodes()+1):
                            p=tri.Node(n).Transformed(trsf);vertices.append([p.X(),p.Y(),p.Z()])
                        for n in range(1,tri.NbTriangles()+1):
                            a,b,c=tri.Triangle(n).Get()
                            if face.Orientation()==TopAbs_REVERSED:b,c=c,b
                            faces.append([start+a-1,start+b-1,start+c-1])
                    explorer.Next()
                if not faces:return
                mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
                # Keep tessellated topology: aggressive assembly decimation
                # distorted thin rails and wheels in the original preview.
                mesh.export(meshfile)
                color=Quantity_Color();rgba=[.65,.68,.71,1]
                for kind in [XCAFDoc_ColorType.XCAFDoc_ColorSurf,XCAFDoc_ColorType.XCAFDoc_ColorGen]:
                    if colors.GetColor_s(definition,kind,color):
                        rgba=[color.Red(),color.Green(),color.Blue(),1];break
                meta=dict(id=key,name=title,mesh=str(meshfile.relative_to(ROOT)),bounds_mm=mesh.bounds.tolist(),
                          faces=len(mesh.faces),rgba=rgba,mass_kg=None,connection_points=None,legal_vrc=None)
                metafile.write_text(json.dumps(meta,indent=2))
            definitions[key]=meta
            if len(definitions)%10==0:print(name+f': {len(definitions)} part meshes cached',flush=True)
        instances.append(dict(id=f'i{len(instances):05d}',definition=key,path=path,transform_mm=world.tolist()))

    for i in range(1,roots.Length()+1):walk(roots.Value(i),np.eye(4),[])
    all_bounds=[]
    for item in instances:
        lo,hi=np.array(definitions[item['definition']]['bounds_mm'])
        corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        all_bounds.extend((corners@np.array(item['transform_mm']).T)[:,:3])
    bounds=np.array(all_bounds)
    data=dict(version='xcaf-assembly-1',source_sha256=source_hash,
              units='mm (STEP reader target unit)',definitions=definitions,instances=instances,
              bounds_mm=[bounds.min(axis=0).tolist(),bounds.max(axis=0).tolist()],elapsed_s=time.monotonic()-started)
    (out.parent/'assembly_tree.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k not in ['definitions','instances']}),flush=True)
    print(f'{len(definitions)} definitions, {len(instances)} instances',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['clawbot','override_field'])
    import_tree(p.parse_args().name)
