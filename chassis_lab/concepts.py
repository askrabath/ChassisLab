"""AI concept gallery. Schema checks and static rendering ONLY; no simulation."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET
import numpy as np
from pydantic import BaseModel,ConfigDict,Field
from .physics import element

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class Chassis(Strict):
    layout:Literal['rectangle','u_front','h_frame','twin_pod']
    drive:Literal['traction','mixed_omni','holonomic','tracks']
    length_mm:float=Field(ge=240,le=450)
    width_mm:float=Field(ge=220,le=430)
    wheelbase_mm:float=Field(ge=160,le=360)
    wheel_diameter_mm:Literal[69.85,82.55,101.6]
    wheel_count:Literal[4,6,8]
    battery_xy_mm:list[float]=Field(min_length=2,max_length=2)
    gearing_rpm:Literal[100,200,600]

class Shape(Strict):
    name:str
    shape:Literal['box','cylinder','capsule','sphere']
    center_mm:list[float]=Field(min_length=3,max_length=3)
    size_mm:list[float]=Field(min_length=3,max_length=3,description='Full local x/y/z size; cylinder axis is local Z, x=y=diameter; capsule z is total end-to-end length.')
    rotation_deg:list[float]=Field(min_length=3,max_length=3)
    role:Literal['structure','moving_link','intake','guide','motor']

class Concept(Strict):
    name:str=Field(max_length=65)
    chassis:Chassis
    mechanism_family:str=Field(max_length=65)
    scoring_sequence:str=Field(max_length=650)
    distinguishing_idea:str=Field(max_length=500)
    main_uncertainty:str=Field(max_length=350)
    parts:list[Shape]=Field(min_length=8,max_length=22)

class Batch(Strict):
    concepts:list[Concept]=Field(min_length=5,max_length=5)

PROMPT='''Generate five genuinely distinct VEX-inspired Override robot CONCEPTS, each including a different base chassis configuration AND a mechanically different scoring mechanism. This is divergent ideation, not optimization or a physics experiment. Do not claim originality worldwide, legality, fabrication readiness, or measured performance. Use V5-style metal structure, axles, motors, gears, rollers, belts, links and guides. Consider both upright/lying Pins and Cups, acquisition, orientation, lifting, and controlled placement onto a Goal. Pin nominal overall scale ~165 mm and flange diameter ~80 mm; cups similar size; goals at several heights around 83/147/223 mm. Only approximate geometry is available.
Return 8-22 STATIC visual primitives per mechanism, in an informative deployed pose, mounted plausibly to the chassis. Express a recognizable assembly, not a row of disconnected boxes. Coordinates: robot center x=y=0; X forward, Y left, Z up; ground z=0. Wheel centers z=wheel radius. Chassis rails center z=wheel radius+25 mm. Mechanism parts use GLOBAL mm positions, rotations XYZ degrees. Cylinders local Z axis, x=y diameter, z length. Dimensions must be positive 5..600 mm; centers within +/-600 mm horizontally, z=30..750 mm. Omit chassis rails, wheels and battery from parts (renderer adds those). Add visible motors and supporting structure. The red moving links and turquoise rollers should clearly communicate the concept. Include enough linked members to distinguish the idea: e.g. if a linkage or elevator is described, depict its links or mast/carriage. Use varied lift kinematics, acquisition tools and transfer arrangements, not twenty claw-arm variants. Base dimensions describe rail envelope; wheelbase must be below length minus 40 mm. Battery position must lie well inside chassis. Do not invent test results. Keep descriptions concise. These drawings are previews only; joints/controllers have NOT been implemented.'''

def save(path,value):path.write_text(json.dumps(value,indent=2))

def generate(root):
    from dotenv import dotenv_values
    from openai import OpenAI
    # Explicitly follow this project's previously authorized local env source.
    key=dotenv_values('.env.local').get('OPENAI_API_KEY')
    if not key:raise RuntimeError('Project .env.local OPENAI_API_KEY missing')
    client=OpenAI(api_key=key,max_retries=0,timeout=180)
    accepted=[];calls=json.loads((root/'api_calls.json').read_text()) if (root/'api_calls.json').exists() else []
    save(root/'schema.json',Batch.model_json_schema())
    for batch in range(4):
        path=root/f'batch_{batch+1}.json'
        if path.exists():accepted.extend(Batch.model_validate_json(path.read_text()).concepts);continue
        previous=[dict(name=c.name,family=c.mechanism_family,idea=c.distinguishing_idea) for c in accepted]
        messages=[dict(role='system',content=PROMPT),dict(role='user',content=f'Batch {batch+1} of four. Propose five NEW designs. Avoid duplicating any of these already accepted ideas: '+json.dumps(previous))]
        for attempt in range(2):
            if len(calls)>=8:raise RuntimeError('Concept gallery API call budget exhausted')
            record=dict(model='gpt-5.6-sol',reasoning='low',batch=batch+1,attempt=attempt+1,max_output_tokens=10000,input=messages.copy())
            response=None
            try:
                print(f'AI batch {batch+1}/4, attempt {attempt+1}: requesting five concepts',flush=True)
                response=client.responses.parse(model='gpt-5.6-sol',reasoning={'effort':'low'},input=messages,text_format=Batch,max_output_tokens=10000,store=False)
                record.update(response_id=response.id,returned_model=response.model,status=response.status,usage=response.usage.model_dump() if response.usage else None)
                save(root/f'response_{batch+1}_{attempt+1}.json',response.model_dump(mode='json',warnings=False))
                proposal=response.output_parsed
                if proposal is None:raise ValueError('No complete structured proposal')
                for c in proposal.concepts:
                    if c.chassis.wheelbase_mm>c.chassis.length_mm-40:raise ValueError(c.name+': wheelbase exceeds rail envelope')
                    if any(not np.isfinite(v) or v<5 or v>600 for p in c.parts for v in p.size_mm):raise ValueError('Invalid primitive sizes')
                    if any(not np.isfinite(v) for p in c.parts for v in p.center_mm+p.rotation_deg):raise ValueError('Nonfinite coordinates')
                if len({c.name for c in accepted+proposal.concepts})!=len(accepted)+5:raise ValueError('Duplicate concept names')
                save(path,proposal.model_dump());accepted.extend(proposal.concepts)
                record['outcome']='accepted';calls.append(record);save(root/'api_calls.json',calls)
                print('Accepted: '+', '.join(c.name for c in proposal.concepts),flush=True)
                break
            except Exception as exc:
                # Do not persist arbitrary transport exceptions which could
                # contain authentication headers or other credential material.
                record['outcome']='failed';record['error_type']=type(exc).__name__
                calls.append(record);save(root/'api_calls.json',calls)
                if attempt==1:raise RuntimeError(f'Batch {batch+1} failed: {type(exc).__name__}') from None
                messages.append(dict(role='user',content='The prior output failed schema or geometry validation. Return five complete finite designs satisfying all bounds and no duplicate names.'))
    save(root/'concepts.json',[c.model_dump() for c in accepted])
    save(root/'manifest.json',dict(source='OpenAI Responses API',model='gpt-5.6-sol',reasoning_effort='low',count=len(accepted),physics_steps=0,performance_tests_run=False,status='generated',max_calls=8,max_output_tokens_per_call=10000,description='Untested concept previews; architecture extension separate from evaluated assembly schema.'))
    return accepted

COLORS={'structure':'.65 .7 .76 1','moving_link':'.87 .16 .19 1','intake':'.03 .65 .65 1','guide':'.96 .66 .13 1','motor':'.12 .16 .21 1'}

def compile_preview(c,path):
    root=ET.Element('mujoco',model=c.name);element(root,'compiler',angle='degree')
    visual=element(root,'visual');element(visual,'global',offwidth=800,offheight=600)
    element(visual,'quality',offsamples=0);element(visual,'headlight',ambient='.45 .45 .45')
    world=element(root,'worldbody');element(world,'light',pos='0 -1 3',castshadow='false')
    element(world,'geom',type='plane',size='2 2 .1',rgba='.09 .13 .18 1')
    def geom(shape,pos,size,rgba,euler=(0,0,0)):
        element(world,'geom',type=shape,pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),euler=' '.join(map(str,euler)),rgba=rgba,contype=0,conaffinity=0)
    b=c.chassis;l=b.length_mm/1000;w=b.width_mm/1000;r=b.wheel_diameter_mm/2000;z=r+.025
    for side in [-1,1]:geom('box',[0,side*w/2,z],[l/2,.009,.018],COLORS['structure'])
    bars=[-l/2,l/2] if b.layout=='rectangle' else [-l/2,0] if b.layout=='u_front' else [0]
    for x in bars:geom('box',[x,0,z],[.009,w/2,.018],COLORS['structure'])
    if b.layout=='twin_pod':
        for s in [-1,1]:geom('box',[0,s*(w/2-.035),z],[l/2,.035,.01],COLORS['motor'])
    for side in [-1,1]:
        for x in np.linspace(-b.wheelbase_mm/2000,b.wheelbase_mm/2000,b.wheel_count//2):
            geom('cylinder',[x,side*(w/2+.021),r],[r,.015],'.08 .1 .13 1',(90,0,0))
            geom('cylinder',[x,side*(w/2+.038),r],[r*.48,.003],'.4 .45 .5 1',(90,0,0))
        if b.drive=='tracks':
            for height in [r*.25,r*1.75]:geom('box',[0,side*(w/2+.021),height],[b.wheelbase_mm/2000,.025,.009],'.12 .15 .18 1')
    geom('box',[b.battery_xy_mm[0]/1000,b.battery_xy_mm[1]/1000,z+.025],[.07,.022,.015],COLORS['motor'])
    for p in c.parts:
        size=np.array(p.size_mm)/2000
        if p.shape=='cylinder':size=[size[0],size[2]]
        elif p.shape=='sphere':size=[size[0]]
        elif p.shape=='capsule':size=[min(size[0],size[1]),max(.001,size[2]-min(size[0],size[1]))]
        geom(p.shape,np.array(p.center_mm)/1000,size,COLORS[p.role],p.rotation_deg)
    path.write_text(ET.tostring(root,encoding='unicode'))

def render_gallery(root,concepts):
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
    small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14)
    cards=[];tiles=[]
    for index,c in enumerate(concepts,1):
        folder=root/f'{index:02d}';folder.mkdir(exist_ok=True);save(folder/'design.json',c.model_dump())
        path=folder/'preview.xml';compile_preview(c,path)
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
        # Forward kinematics only. No mj_step, controllers or benchmark calls.
        mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[.06,0,.24];cam.distance=1.5;cam.azimuth=135;cam.elevation=-25
        with mujoco.Renderer(m,height=600,width=800) as renderer:
            renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());im.save(folder/'preview.png')
        tile=Image.new('RGB',(500,440),'#111b27');tile.paste(im.resize((500,375)),(0,0));draw=ImageDraw.Draw(tile)
        draw.text((12,380),f'{index:02d}  {c.name}',font=font,fill='white')
        draw.text((12,410),c.mechanism_family[:62],font=small,fill='#71d9d0');tiles.append(tile)
        esc=html.escape
        cards.append(f'<article><h2>{index:02d} · {esc(c.name)}</h2><a href="{index:02d}/preview.png"><img src="{index:02d}/preview.png"></a><h3>{esc(c.mechanism_family)}</h3><p>{esc(c.distinguishing_idea)}</p><p><b>Scoring sequence:</b> {esc(c.scoring_sequence)}</p><p><b>Chassis:</b> {bdesc(c.chassis)}</p><p><b>Uncertainty:</b> {esc(c.main_uncertainty)}</p><a href="{index:02d}/design.json">AI design JSON</a></article>')
        print(f'Rendered {index:02d}: {c.name}',flush=True)
    sheet=Image.new('RGB',(2000,2200),'#111b27')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*500,(i//4)*440))
    sheet.save(root/'all_20.png')
    for page in range(2):
        smaller=Image.new('RGB',(1000,2200),'#111b27')
        for i,tile in enumerate(tiles[page*10:page*10+10]):smaller.paste(tile,((i%2)*500,(i//2)*440))
        smaller.save(root/f'concepts_{page*10+1:02d}_{page*10+10:02d}.png')
    (root/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>20 AI robot concepts</title><style>body{font:16px system-ui;background:#101924;color:#e8edf4;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:24px}article{background:#1c2938;padding:20px;border-radius:14px}img{width:100%}a{color:#71d9d0}p{line-height:1.5}</style><h1>20 AI chassis + scoring concepts</h1><p>GPT-5.6 Sol · low reasoning · AI-authored JSON → MuJoCo static previews · UNTESTED</p><p>Colors: silver structure, red moving links, turquoise intake, gold guides. Illustrations are schematic; joints, transmissions and hardware feasibility are not verified. No physics steps, scoring trials, rankings or improvement claims.</p><main>'+''.join(cards)+'</main>')

def bdesc(b):return html.escape(f'{b.layout}; {b.drive}; {b.wheel_count} wheels; {b.length_mm:g} × {b.width_mm:g} mm rails; {b.wheel_diameter_mm:g} mm wheels; {b.gearing_rpm} rpm gearing')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--render-only',action='store_true');a=p.parse_args()
    root=Path(a.output or 'concept_runs/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_sol20');root.mkdir(parents=True,exist_ok=True)
    print(root.resolve(),flush=True)
    designs=[Concept.model_validate(c) for c in json.loads((root/'concepts.json').read_text())] if a.render_only else generate(root)
    render_gallery(root,designs)
    manifest=json.loads((root/'manifest.json').read_text());manifest['status']='rendered'
    calls=json.loads((root/'api_calls.json').read_text());manifest['actual_api_calls']=len(calls)
    manifest['token_usage']={key:sum((c.get('usage') or {}).get(key,0) for c in calls) for key in ['input_tokens','output_tokens','total_tokens']}
    save(root/'manifest.json',manifest)
    print('Gallery: '+str((root/'gallery.html').resolve()),flush=True)

if __name__=='__main__':main()
