"""Part-backed assembly variants with traceable display LOD and explicit gaps.

Original CAD placement is evidence of source placement, not proof of mates.
No scaling, procedural robot visual geometry, scoring tests or physics steps.
"""
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict,Field
from .cad_scene import ROOT,tree

class Variant(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str
    driven_wheels:Literal['source_traction','source_omni']
    passive_wheels:Literal['source_traction','source_omni']
    rationale:str
    scoring_mechanism:Literal['unchanged_source_clawbot_arm_and_claw']
    unresolved_build_risks:list[str]

class Four(BaseModel):
    model_config=ConfigDict(extra='forbid')
    designs:list[Variant]=Field(min_length=4,max_length=4)

def save(path,data):path.write_text(json.dumps(data,indent=2))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def catalog(output):
    source=tree('clawbot');parts={}
    for key,d in source['definitions'].items():
        high=ROOT/d['mesh'];low=high.with_name(high.stem+'_lod.obj')
        number=re.search(r'\b27[56]-\d{4}(?:-\d{3})?\b',d['name'])
        parts[key]=dict(source_definition=key,name=d['name'],part_number=number[0] if number else None,
            canonical_step='data/cad_assets/clawbot/canonical.step',canonical_step_sha256=source['source_sha256'],
            high_mesh=str(high.relative_to(ROOT)),high_sha256=digest(high),low_mesh=str(low.relative_to(ROOT)),low_sha256=digest(low),
            placement_units='mm',visual_mesh_units='mm; MJCF uses 0.001 scale',mass_kg=None,verified_connections=None,
            identity_status='source CAD label; not independently verified against manufacturer catalog')
    data=dict(parts=parts,source_placement_instances=source['instances'],
        limitations=['No verified mates, material mass or full fastener/shaft bill of materials supplied.',
        'Part IDs with the same VEX label can be different configurations. Source definition IDs are retained separately.',
        'Low-detail geometry is a display representation only; it never determines mass or collision geometry.'])
    save(output/'part_catalog.json',data);return data

def assemble(spec):
    data=copy.deepcopy(tree('clawbot'));changes=[]
    for item in data['instances']:
        if item['id'] not in {'i00010','i00024','i00013','i00021'}:continue
        selected=spec.driven_wheels if item['id'] in {'i00010','i00024'} else spec.passive_wheels
        target='p00010' if selected=='source_traction' else 'p00012';original=item['definition']
        if target==original:continue
        # Keep the original axle axis and wheel-width center plane. Local mesh
        # origins differ; do not stretch or resize either real CAD wheel.
        old_mid=np.mean(data['definitions'][original]['bounds_mm'],axis=0)[1]
        new_mid=np.mean(data['definitions'][target]['bounds_mm'],axis=0)[1]
        pose=np.array(item['transform_mm']);pose[:3,3]+=pose[:3,1]*(old_mid-new_mid)
        item['transform_mm']=pose.tolist();item['definition']=target
        changes.append(dict(instance=item['id'],from_definition=original,to_definition=target,
            operation='replace wheel at source axle axis; preserve width-center plane',
            connection_status='UNVERIFIED: axle engagement, bearing spacing and spacer stack must be checked'))
    data['variant_name']=spec.name;data['changes']=changes
    data['assembly_status']='source-derived CAD assembly; NOT certified buildable'
    return data

def audit_pair(high,low):
    """Representation consistency checks; not a performance test."""
    import mujoco
    a=mujoco.MjModel.from_xml_path(str(high.resolve()));b=mujoco.MjModel.from_xml_path(str(low.resolve()))
    checks={name:bool(np.array_equal(getattr(a,name),getattr(b,name))) for name in ['body_mass','body_inertia','jnt_axis','jnt_type','actuator_ctrlrange']}
    for name in ['timestep','iterations']:checks[name]=bool(getattr(a.opt,name)==getattr(b.opt,name))
    # Contact-only geometry must not use decimated visual meshes.
    for m in [a,b]:
        for i in range(m.ngeom):
            if m.geom_contype[i] and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,m.geom_dataid[i])
                if name.endswith('visual_lod'):raise ValueError('Visual LOD used for contact')
    if not all(checks.values()):raise ValueError('Visual LOD changed dynamics configuration')
    return dict(representation_checks=checks,physics_steps=0,performance_tests_run=False,buildability_verified=False)

def preview(path,target):
    import mujoco
    from PIL import Image
    m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera();cam.lookat[:]=d.qpos[:3]+[.03,0,.09];cam.distance=1.0;cam.azimuth=135;cam.elevation=-25
    with mujoco.Renderer(m,height=600,width=800) as r:
        r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(target)

def build(output,specs):
    from .cad_drive import compile_scene
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cat=catalog(output)
    for index,spec in enumerate(specs,1):
        folder=output/f'{index:02d}';folder.mkdir(exist_ok=True);data=assemble(spec)
        save(folder/'design.json',spec.model_dump());save(folder/'assembly.json',data)
        counts=Counter(i['definition'] for i in data['instances'])
        save(folder/'bom.json',[dict(definition=k,name=cat['parts'][k]['name'],part_number=cat['parts'][k]['part_number'],quantity=v) for k,v in counts.items()])
        high,_=compile_scene(folder/'high',lightweight=False,crop=True,robot_data=data)
        low,_=compile_scene(folder/'low',lightweight=True,crop=True,robot_data=data)
        save(folder/'audit.json',audit_pair(high,low))
        preview(high,folder/'high.png');preview(low,folder/'low.png')
        print(f'{index}: {spec.name}: original CAD + low-detail model rendered',flush=True)
    import html
    from PIL import Image,ImageDraw,ImageFont
    sheet=Image.new('RGB',(1600,1320),'#14202c');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',22)
    cards=[]
    for index,spec in enumerate(specs,1):
        folder=output/f'{index:02d}';x=((index-1)%2)*800;y=((index-1)//2)*660
        image=Image.open(folder/'high.png').resize((800,600));sheet.paste(image,(x,y))
        draw.text((x+15,y+607),f'{index:02d} {spec.name}',font=font,fill='white')
        draw.text((x+15,y+637),'Source CAD parts · same arm/claw · untested',font=font,fill='#82d8d0')
        esc=html.escape
        cards.append(f'<article><h2>{index:02d} · {esc(spec.name)}</h2><p>{esc(spec.rationale)}</p><div class="pair"><figure><img src="{index:02d}/high.png"><figcaption>Original CAD detail</figcaption></figure><figure><img src="{index:02d}/low.png"><figcaption>Reduced display detail; same physics parameters</figcaption></figure></div><p>Driven axle wheels: {esc(spec.driven_wheels)}; passive axle wheels: {esc(spec.passive_wheels)}.</p><p><b>Unverified:</b> {esc("; ".join(spec.unresolved_build_risks))}</p><p><a href="{index:02d}/assembly.json">Assembly & source placements</a> · <a href="{index:02d}/bom.json">Source bill of materials</a> · <a href="{index:02d}/audit.json">Representation checks</a></p></article>')
    sheet.save(output/'four_designs.png')
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Four source-part assemblies</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;background:#14202c;color:#eee}article{background:#20303f;padding:20px;margin:20px 0;border-radius:12px}.pair{display:flex}figure{width:50%;margin:10px}img{width:100%}a{color:#82d8d0}</style><h1>Four catalog-constrained Clawbot variants</h1><p>All robot visual parts come from the supplied Clawbot CAD. These are wheel-configuration variants with the SAME scoring mechanism, not four novel mechanisms. No simulation tests were run.</p><p>Source placement is retained except declared wheel swaps. CAD has no verified mates or complete shaft/fastener inventory; these are not certified buildable. Physics masses, wheel friction and locked-arm contacts remain approximate. No robot visual geometry was fabricated from primitive concept sketches.</p>'+''.join(cards))
    save(output/'manifest.json',dict(designs=4,source='OpenAI Responses API restricted to source-part wheel variants',physics_steps=0,performance_tests_run=False,
        high_detail_source='supplied Clawbot canonical STEP and per-definition tessellations',
        design_scope='four wheel configurations; identical source arm/claw',buildability_verified=False))

def generate(output):
    from openai import OpenAI
    from dotenv import dotenv_values
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'ai_designs.json'
    if path.exists():return Four.model_validate_json(path.read_text()).designs
    key=dotenv_values('.env.local').get('OPENAI_API_KEY')
    if not key:raise RuntimeError('Project API credential missing')
    prompt='''Propose exactly four conservative variants of the supplied VEX Clawbot CAD assembly. This is a REAL-PART representation experiment, NOT four novel mechanisms. Use each combination of driven_wheels / passive_wheels (source_traction, source_omni) exactly once. These labels select the existing 4-inch traction wheel CAD and existing 276-2185 omni wheel CAD, not invented wheel geometry. Keep the entire original chassis, motor locations, battery, lift transmission, and claw mechanism. Give clear names and explain possible tradeoffs as hypotheses, without testing or claiming improvement. Explicitly flag unknown axle length/engagement, spacers, ground-clearance and source assembly hardware completeness. The catalog contains no verified mating constraints, shaft BOM or calibrated mass properties. Do not claim buildability or legality. These four modest drivetrain variants are a starting point for source-CAD -> lower-detail display + separate physics, not a complete parts-search system.'''
    client=OpenAI(api_key=key,max_retries=0,timeout=90)
    response=client.responses.parse(model='gpt-5.6-sol',reasoning={'effort':'low'},input=prompt,text_format=Four,max_output_tokens=3500,store=False)
    save(output/'api_record.json',dict(model=response.model,reasoning='low',response_id=response.id,input=prompt,usage=response.usage.model_dump() if response.usage else None,status=response.status))
    result=response.output_parsed
    if result is None or len({(d.driven_wheels,d.passive_wheels) for d in result.designs})!=4:raise ValueError('AI did not return four distinct supported configurations')
    save(path,result.model_dump());return result.designs

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='part_design_runs/four_catalog_designs');p.add_argument('--render-only',action='store_true');a=p.parse_args()
    specs=Four.model_validate_json(Path(a.output,'ai_designs.json').read_text()).designs if a.render_only else generate(a.output)
    build(a.output,specs)

if __name__=='__main__':main()
