"""AI-selected mechanism layouts using unscaled source CAD, with no simulation.

These are layout concepts, not solved mechanical assemblies. Explicitly preserve
that distinction in JSON, MJCF names and every preview.
"""
import copy
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .cad_scene import ROOT, tree, scene_base, add_visual
from .physics import element
from .part_assemblies import save

FAMILIES=('four_bar_fork','roller_feeder','tilting_tray','overhead_claw')

class Proposal(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str
    mechanism:Literal['four_bar_fork','roller_feeder','tilting_tray','overhead_claw']
    display_angle_deg:float=Field(ge=15,le=60)
    lateral_spacing_mm:Literal[101.6,127.0,152.4]
    rationale:str
    scoring_sequence:list[str]
    hypothesis:str
    required_parts:list[str]
    unresolved_connections:list[str]

class Four(BaseModel):
    model_config=ConfigDict(extra='forbid')
    designs:list[Proposal]=Field(min_length=4,max_length=4)
    @model_validator(mode='after')
    def distinct(self):
        if {d.mechanism for d in self.designs}!=set(FAMILIES):
            raise ValueError('Use each of the four different mechanism families once')
        return self

def generate(out):
    from dotenv import dotenv_values
    from openai import OpenAI
    catalog=json.loads((ROOT/'data/official_catalog/catalog.json').read_text())
    available=[dict(sku=p['sku'],name=p['name'],cad_available=bool(p['cad'])) for p in catalog['parts'].values()]
    prompt='''Generate four NEW VEX-inspired Override scoring concepts, one per mechanism family:
four_bar_fork (parallel linkage lifts two metal fork rails), roller_feeder (powered wheels feed objects along an inclined guide), tilting_tray (a pivoting channel cradle dumps objects), overhead_claw (raised cantilever arm with the source claw).
The base chassis is the same supplied four-wheel Clawbot base; mechanisms must differ in how they acquire/place objects, not wheel choices. A deterministic layout compiler will arrange unscaled real CAD channels, wheels, motors and the source claw, using your angle and spacing. These are CONCEPT LAYOUTS, not mating-solved assemblies. Missing shafts, collars, fasteners, transmission, support and object retention must be explicitly called out. Do not claim buildability, official robot legality, successful scoring or testing. Required_parts must contain ONLY exact SKUs from the attached official list; include necessary bearings, shafts, collars, screws and transmission parts even when their CAD is missing. Limit to 12 SKUs per proposal and 3 short scoring steps. Available source CAD includes configured 276-7285 channels (254/190.5/63.5mm), 276-2304 angle, 276-4842 motor, 276-2185 wheel, 276-3438 gear, 276-1209 bearing, source Clawbot claw subassembly whose kit mapping is unverified. Display angle is a static pose, not a proven operating range. Give specific physical uncertainties, not generic disclaimers. Do not assert the benchmark field goal openings have been calibrated. No performance simulations will be run.
Official legal inventory (a listed SKU does not certify the assembled robot):
'''+json.dumps(available)
    save(out/'proposal_schema.json',Four.model_json_schema())
    client=OpenAI(api_key=dotenv_values(ROOT/'.env.local').get('OPENAI_API_KEY'),timeout=150,max_retries=0)
    for attempt in range(2):
        try:
            r=client.responses.parse(model='gpt-5.6-sol',reasoning={'effort':'low'},input=prompt,text_format=Four,max_output_tokens=5000,store=False)
            save(out/f'api_record_{attempt}.json',dict(model=r.model,response_id=r.id,status=r.status,usage=r.usage.model_dump() if r.usage else None,prompt=prompt,proposal=r.output_parsed.model_dump() if r.output_parsed else None))
            if r.output_parsed is None:raise ValueError('Missing structured response')
            result=Four.model_validate(r.output_parsed.model_dump())
            for d in result.designs:
                missing=set(d.required_parts)-catalog['parts'].keys()
                if missing:raise ValueError('Unknown SKUs: '+str(sorted(missing)))
            save(out/'ai_designs.json',result.model_dump());return result
        except Exception as e:
            # API exception text may contain sensitive transport data; record type only.
            save(out/f'failure_{attempt}.json',dict(error_type=type(e).__name__,validation_error=str(e) if isinstance(e,ValueError) else None))
            if attempt:raise RuntimeError('Bounded proposal generation failed; inspect failure records') from None
            prompt+='\nRepair: return all four distinct families and only exact listed SKUs. '+(str(e) if isinstance(e,ValueError) else '')

def assembly(spec):
    source=tree('clawbot');data=copy.deepcopy(source);data['instances']=[]
    # Source chassis is the first 26 instances. Remove the entire old manipulator.
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    wheels=[i for i in source['instances'] if i['id'] in {'i00010','i00013','i00021','i00024'}]
    center=np.mean([np.array(i['transform_mm'])[:3,3] for i in wheels],axis=0)
    world=np.eye(4);world[:3,:3]=rotation;world[:3,3]=-rotation@center+[0,0,53.1]
    for item in source['instances'][:26]:
        i=copy.deepcopy(item);i['transform_mm']=(world@np.array(i['transform_mm'])).tolist();i['role']='source_chassis';data['instances'].append(i)
    def place(key,position,rot=None,role='mechanism'):
        rot=np.eye(3) if rot is None else np.array(rot)
        mid=np.mean(source['definitions'][key]['bounds_mm'],axis=0)
        p=np.eye(4);p[:3,:3]=rot;p[:3,3]=np.array(position)-rot@mid
        data['instances'].append(dict(id=f'new{len(data["instances"]):04d}',definition=key,transform_mm=p.tolist(),role=role))
    def ry(a):
        c,s=math.cos(a),math.sin(a);return np.array([[c,0,-s],[0,1,0],[s,0,c]])
    a=math.radians(spec.display_angle_deg);gap=spec.lateral_spacing_mm
    vertical=ry(math.pi/2)
    if spec.mechanism=='four_bar_fork':
        for y in [-gap/2,gap/2]:
            place('p00015',[-90,y,174],vertical,'fixed_tower')
            for z in [150,226]:
                start=np.array([-90,y,z]);end=start+np.array([241.3*math.cos(a),0,241.3*math.sin(a)])
                place('p00019',(start+end)/2,ry(a),'parallel_lift_link')
                for point in [start,end]:place('p00003',point,role='pivot_bearing_layout')
            x=-90+241.3*math.cos(a);z=188+241.3*math.sin(a)
            place('p00023',[x,y,z],vertical,'upright_coupler')
            place('p00015',[x+82,y,z-38],role='fork')
            place('p00002',[-90,y,128],role='lift_motor_layout')
    elif spec.mechanism=='roller_feeder':
        for y in [-gap/2-30,gap/2+30]:place('p00019',[185,y,170],ry(a),'inclined_side_rail')
        # Source omni wheels are 106.1 mm across: keep adjacent wheels apart.
        for offset in [-114.3,0,114.3]:
            x=185+offset*math.cos(a);z=170+offset*math.sin(a)
            for y in [-gap/2,gap/2]:place('p00012',[x,y,z+38],role='feed_wheel')
            for y in [-gap/2-30,gap/2+30]:place('p00003',[x,y,z+38],role='shaft_bearing_layout')
        for y in [-gap/2-32,gap/2+32]:place('p00002',[185,y,100],role='feed_motor_layout')
        for y in [-gap/2,gap/2]:place('p00015',[70,y,130],vertical,'support')
    elif spec.mechanism=='tilting_tray':
        for y in [-gap/2,gap/2]:
            place('p00015',[0,y,166],vertical,'pivot_support')
            place('p00003',[0,y,244],role='tray_pivot_bearing_layout')
        for y in np.linspace(-gap/2,gap/2,5):place('p00019',[116*math.cos(a),y,244+116*math.sin(a)],ry(a),'tray_slats')
        place('p00015',[0,0,244],[[0,-1,0],[1,0,0],[0,0,1]],'tray_crossmember')
        place('p00002',[0,-gap/2-28,210],role='tray_motor_layout')
        place('p00022',[0,-gap/2-18,244],role='tray_gear_layout')
    else:
        for y in [-50.8,50.8]:place('p00019',[-100,y,205],vertical,'mast')
        start=np.array([-100,0,329]);end=start+[241.3*math.cos(a),0,241.3*math.sin(a)]
        place('p00019',(start+end)/2,ry(a),'overhead_boom')
        place('p00002',[-95,0,342],role='boom_motor_layout')
        claw=[i for i in source['instances'] if any('V5 Claw V2 Assembly' in p for p in i['path'])]
        # Preserve all source claw internal relative placements, including washers.
        poses=[world@np.array(i['transform_mm']) for i in claw]
        mid=np.mean([p[:3,3] for p in poses],axis=0)
        for item,p in zip(claw,poses):
            p[:3,3]+=end+[70,0,0]-mid
            i=copy.deepcopy(item);i['id']='claw_'+i['id'];i['transform_mm']=p.tolist();i['role']='source_claw';data['instances'].append(i)
    data['status']='UNTESTED_CONCEPT_LAYOUT_WITH_UNVERIFIED_CONNECTIONS'
    data['compiler']='Hand-authored mechanism layout templates; AI selects family parameters and rationale'
    return data

def render(data,out,low=False):
    import xml.etree.ElementTree as ET
    import mujoco
    from PIL import Image
    root,asset,world=scene_base();root.set('model','static_concept_NOT_physics_ready')
    element(world,'geom',type='plane',size='2 2 .1',rgba='.12 .16 .21 1')
    for i in data['instances']:
        d=copy.deepcopy(data['definitions'][i['definition']])
        if low:
            path=ROOT/d['mesh'];p=path.with_name(path.stem+'_lod.obj')
            if p.exists():d['mesh']=str(p.relative_to(ROOT))
        # Small hardware remains visible in BOTH representations.
        add_visual(asset,world,i,d,'concept')
    name='low' if low else 'high';xml=out/(name+'.xml');xml.write_text(ET.tostring(root,encoding='unicode'))
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.065,0,.26];camera.distance=1.23;camera.azimuth=135;camera.elevation=-24
    with mujoco.Renderer(m,height=750,width=1000) as renderer:
        renderer.update_scene(d,camera=camera);Image.fromarray(renderer.render()).save(out/(name+'.png'))
    return dict(physics_steps=0,visible_instances=len(data['instances']),mesh_assets=m.nmesh,simulation_ready=False)

def build(out,result):
    from collections import Counter
    from PIL import Image,ImageDraw,ImageFont
    cat=json.loads((ROOT/'data/official_catalog/catalog.json').read_text());cards=[]
    sheet=Image.new('RGB',(2000,1660),'#14202c');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',25)
    for n,spec in enumerate(result.designs,1):
        folder=out/f'{n:02d}';folder.mkdir(exist_ok=True);data=assembly(spec)
        save(folder/'design.json',spec.model_dump());save(folder/'assembly.json',data)
        counts=Counter(i['definition'] for i in data['instances'])
        save(folder/'rendered_bom.json',[dict(definition=k,name=data['definitions'][k]['name'],quantity=v,mesh=data['definitions'][k]['mesh']) for k,v in counts.items()])
        save(folder/'required_parts.json',[cat['parts'][sku] for sku in spec.required_parts])
        audit=render(data,folder);render(data,folder,low=True);save(folder/'preview_status.json',audit)
        x=((n-1)%2)*1000;y=((n-1)//2)*830;sheet.paste(Image.open(folder/'high.png'),(x,y))
        draw.text((x+15,y+759),f'{n:02d} {spec.name}',font=font,fill='white')
        draw.text((x+15,y+793),'CAD layout concept · connections unverified · NOT tested',font=font,fill='#ffcf83')
        e=html.escape
        caveat='<p><b>CAD coverage gap:</b> The feeder displays available source omni wheels. The AI requests flex-wheel hardware; the visible wheels are layout stand-ins, not a physically equivalent assembly.</p>' if spec.mechanism=='roller_feeder' else ''
        cards.append(f'<article><h2>{n:02d} {e(spec.name)}</h2><p>{e(spec.mechanism)} · {e(spec.rationale)}</p><img src="{n:02d}/high.png">{caveat}<details><summary>Reduced display detail (all hardware retained)</summary><img src="{n:02d}/low.png"></details><p>{e(" → ".join(spec.scoring_sequence))}</p><p>Hypothesis: {e(spec.hypothesis)}</p><p><b>Unresolved:</b> {e("; ".join(spec.unresolved_connections))}</p><a href="{n:02d}/assembly.json">Part placements</a> · <a href="{n:02d}/rendered_bom.json">Rendered BOM</a> · <a href="{n:02d}/required_parts.json">AI required parts (includes unrendered hardware)</a></article>')
        print(f'Rendered {n}: {spec.name}; zero physics steps',flush=True)
    sheet.save(out/'four_mechanisms.png')
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Four different scoring concepts</title><style>body{font:16px system-ui;background:#14202c;color:#eee;max-width:1250px;margin:30px auto}article{padding:20px;background:#20303f;margin:20px 0;border-radius:12px}img{max-width:100%}a{color:#86dfd4}</style><h1>Four different scoring mechanism concepts</h1><p>Actual OpenAI proposals, compiled through hand-authored layout templates using unscaled source CAD. Same base chassis; different scoring mechanisms. These are static layouts, NOT build-ready or physics-ready robots. Shafts, fastener stacks, transmissions and clearances are not solved; required and rendered BOMs are separate.</p><p>All 418 legal-list SKUs are cataloged; CAD coverage is incomplete. No scoring tests or physics steps were run.</p><a href="../../data/official_catalog/index.html">Parts catalog</a> · <a href="ai_designs.json">Original AI proposals</a>'+''.join(cards))
    save(out/'manifest.json',dict(designs=4,distinct_mechanisms=list(FAMILIES),ai_model='gpt-5.6-sol',reasoning='low',physics_steps=0,buildability_verified=False,simulation_ready=False,catalog_sha256=hashlib.sha256((ROOT/'data/official_catalog/catalog.json').read_bytes()).hexdigest()))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='part_design_runs/four_new_mechanisms');p.add_argument('--render-only',action='store_true');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    result=Four.model_validate_json((out/'ai_designs.json').read_text()) if a.render_only else generate(out)
    build(out,result)

if __name__=='__main__':main()
