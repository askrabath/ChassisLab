"""Quick part-level V5-style chassis assembly; no API or search loop required.

Bolted components are named fixed child bodies. Shaft/wheel assemblies have hinge
joints. Visual perforations and fasteners are distinct from simple collision
proxies; screw threads and bearing clearance are not contact simulations.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .physics import element

PITCH = 0.0127
RADIUS = 0.0508
THICKNESS = 0.0016
HOLE = 0.0046
OUTPUT = 'assemblies/standard_v5'
SILVER = '0.65 0.69 0.74 1'
DARK = '0.12 0.14 0.17 1'
SOURCES = [
    'https://kb.vex.com/hc/en-us/articles/360035953131-Designing-a-V5-Chassis',
    'https://kb.vex.com/hc/en-us/articles/360035591372-Using-V5-Shafts',
    'https://kb.vex.com/hc/en-us/articles/360035952791-Using-V5-Fasteners',
    'https://www.vexrobotics.com/v5-structure.html',
]


def vec(values):
    return ' '.join(f'{v:.9g}' for v in values)


def add_box_mesh(vertices, faces, center, half):
    offset=len(vertices)
    for z in (-1,1):
        for y in (-1,1):
            for x in (-1,1):
                vertices.append([center[0]+x*half[0], center[1]+y*half[1], center[2]+z*half[2]])
    for a,b,c,d in [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]:
        faces.extend([[offset+a,offset+b,offset+c],[offset+a,offset+c,offset+d]])


def perforated_panel(vertices, faces, columns, rows, origin, axes):
    """Build a sheet with real square visual holes, using four strips per cell."""
    for i in range(columns):
        for j in range(rows):
            a=(i-(columns-1)/2)*PITCH; b=(j-(rows-1)/2)*PITCH
            border=(PITCH-HOLE)/2
            rectangles=[(a-(PITCH+HOLE)/4,b,border/2,PITCH/2),
                        (a+(PITCH+HOLE)/4,b,border/2,PITCH/2),
                        (a,b-(PITCH+HOLE)/4,HOLE/2,border/2),
                        (a,b+(PITCH+HOLE)/4,HOLE/2,border/2)]
            for u,v,hu,hv in rectangles:
                center=list(origin); half=[THICKNESS/2]*3
                center[axes[0]]+=u;center[axes[1]]+=v
                half[axes[0]]=hu;half[axes[1]]=hv
                add_box_mesh(vertices,faces,center,half)


def channel_mesh(asset):
    vertices=[];faces=[]
    perforated_panel(vertices,faces,25,3,[0,0,0],(0,2))
    for sign in (-1,1):
        perforated_panel(vertices,faces,25,1,[0,PITCH/2,sign*(1.5*PITCH-THICKNESS/2)],(0,1))
    element(asset,'mesh',name='perforated_channel',vertex=vec(np.array(vertices).ravel()),face=' '.join(str(v) for f in faces for v in f))


def build():
    root=ET.Element('mujoco',model='standard_v5_part_assembly')
    element(root,'compiler',angle='radian',inertiafromgeom='true',inertiagrouprange='0 5')
    element(root,'option',timestep=.002,gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic')
    default=element(root,'default')
    element(default,'geom',friction='0.7 0.002 0.0001',solref='0.01 1',condim=3)
    visual=element(root,'visual');element(visual,'global',offwidth=1400,offheight=1000)
    element(visual,'headlight',ambient='0.45 0.45 0.45',diffuse='0.7 0.7 0.7',specular='0.15 0.15 0.15')
    asset=element(root,'asset');channel_mesh(asset)
    element(asset,'texture',name='tiles',type='2d',builtin='checker',rgb1='.24 .28 .32',rgb2='.29 .33 .37',width=512,height=512)
    element(asset,'material',name='floor_material',texture='tiles',texrepeat='8 8',reflectance='.05')
    world=element(root,'worldbody')
    element(world,'light',pos='0 -1 2',dir='0 0 -1',diffuse='.8 .8 .8')
    element(world,'geom',name='floor',type='plane',size='2 2 .1',material='floor_material')
    chassis=element(world,'body',name='chassis',pos=f'0 0 {RADIUS+.003}')
    element(chassis,'freejoint',name='root')
    parts=[];connections=[]

    def part(parent,name,kind,pos=(0,0,0),**attrs):
        body=element(parent,'body',name=name,pos=vec(pos),**attrs)
        parts.append(dict(name=name,type=kind))
        if parent is not chassis:
            connections.append(dict(part=name,to=parent.get('name'),type='fixed'))
        return body

    def cosmetic(body,name,type='box',**attrs):
        return element(body,'geom',name=name,type=type,mass=0,contype=0,conaffinity=0,**attrs)

    def bolt(parent,name,pos,axis='z',nut=True,length=.009525):
        b=part(parent,name,'8-32 screw and nut' if nut else '8-32 motor screw',pos)
        orientation={} if axis=='z' else dict(quat='.707106781 .707106781 0 0')
        element(b,'geom',name=name+'_shank',type='cylinder',size=f'.00208 {length/2}',mass=.0015,contype=0,conaffinity=0,rgba=SILVER,**orientation)
        delta=np.array([0,0,length/2+.0014]) if axis=='z' else np.array([0,length/2+.0014,0])
        cosmetic(b,name+'_head',type='cylinder',pos=vec(delta),size='.0038 .0014',rgba=DARK,**orientation)
        cosmetic(b,name+'_socket',type='cylinder',pos=vec(delta*1.2),size='.0013 .0003',rgba='0.02 0.02 0.02 1',**orientation)
        if nut:
            cosmetic(b,name+'_nut',pos=vec(-delta),size='.0034 .0034 .0015' if axis=='z' else '.0034 .0015 .0034',rgba=SILVER)
        return name

    def channel(name,pos,angle=0):
        b=part(chassis,name,'1x3x1x25 C-channel',pos,euler=f'0 0 {angle}')
        # Flanges open along local +y. A 180-degree yaw flips the opening.
        cosmetic(b,name+'_perforations',type='mesh',mesh='perforated_channel',rgba=SILVER)
        element(b,'geom',name=name+'_web_collision',type='box',size=f'{12.5*PITCH} {THICKNESS/2} {1.5*PITCH}',mass=.050,group=3)
        for sign in (-1,1):
            element(b,'geom',name=f'{name}_flange_{sign}',type='box',pos=f'0 {PITCH/2} {sign*(1.5*PITCH-THICKNESS/2)}',size=f'{12.5*PITCH} {PITCH/2} {THICKNESS/2}',mass=.0175,group=3)
        return b

    rails={}
    for side,sign in [('left',1),('right',-1)]:
        for which,y,opening in [('inner',7.5*PITCH,-sign),('outer',11.5*PITCH,sign)]:
            name=f'{side}_{which}_rail'
            rails[name]=channel(name,(0,sign*y,0),angle=0 if opening==1 else math.pi)
            connections.append(dict(part=name,to='frame_crossmembers',type='bolted',fasteners=f'{name}_front_bolt, {name}_rear_bolt'))
    for end,sign in [('front',1),('rear',-1)]:
        # Standard 1.5-inch standoffs raise these full-width crossmembers clear
        # of the tire swept volume (the wheels sit between inner/outer rails).
        channel(f'{end}_crossmember',(sign*10.5*PITCH,0,6*PITCH),angle=-sign*math.pi/2)
        for side,s in [('left',1),('right',-1)]:
            for which,y in [('inner',7*PITCH),('outer',12*PITCH)]:
                name=f'{side}_{which}_{end}_standoff'
                standoff=part(chassis,name,'1.5 inch 8-32 standoff',(sign*11*PITCH,s*y,3*PITCH))
                element(standoff,'geom',name=name+'_metal',type='cylinder',size=f'.003175 {1.5*PITCH}',mass=.008,rgba=SILVER)
                bolt(chassis,f'{side}_{which}_rail_{end}_bolt',(sign*11*PITCH,s*y,1.5*PITCH),nut=False,length=.009525)
                bolt(chassis,name+'_upper_bolt',(sign*11*PITCH,s*y,4.5*PITCH),nut=False,length=.009525)
                connections.append(dict(part=name,to=[f'{side}_{which}_rail',f'{end}_crossmember'],type='two-ended bolted standoff'))

    # Two bolted mounting strips bridge the inner rails and carry the electronics.
    for i,x in enumerate([-.0508,.0508]):
        strip=part(chassis,f'electronics_strip_{i}','perforated mounting strip',(x,0,1.5*PITCH+THICKNESS/2))
        element(strip,'geom',name=f'strip_{i}',type='box',size='.00635 .10795 .0008',mass=.012,rgba=SILVER)
        for s in (-1,1):
            bolt(chassis,f'strip_{i}_bolt_{s}',(x,s*7*PITCH,1.5*PITCH+.001))
        connections.append(dict(part=f'electronics_strip_{i}',to='inner_rails',type='bolted',fasteners=2))
    battery=part(chassis,'battery','V5 battery envelope',(-.060,0,.039))
    element(battery,'geom',name='battery_case',type='box',size='.050 .0325 .01835',mass=.350,rgba='.18 .19 .21 1')
    cosmetic(battery,'battery_label',pos='0 0 .0186',size='.030 .025 .0003',rgba='.8 .12 .13 1')
    for s in (-1,1):
        cosmetic(battery,f'battery_strap_{s}',pos=f'{s*.032} 0 .020',size='.004 .034 .001',rgba=DARK)
    connections.append(dict(part='battery',to='electronics_strip_0',type='rigid strap mount'))
    brain=part(chassis,'brain','V5 brain envelope',(.045,0,.039))
    element(brain,'geom',name='brain_case',type='box',size='.044 .055 .01835',mass=.280,rgba='.79 .11 .14 1')
    cosmetic(brain,'screen_bezel',pos='0 0 .019',size='.035 .042 .0015',rgba='.1 .11 .13 1')
    cosmetic(brain,'screen',pos='0 0 .0206',size='.028 .035 .0002',rgba='.15 .47 .58 1')
    connections.append(dict(part='brain',to='electronics_strip_1',type='bolted',fasteners=['brain_bolt_0','brain_bolt_1']))
    for i,s in enumerate((-1,1)):
        bolt(chassis,f'brain_bolt_{i}',(.0508,s*.045,.023),nut=False)

    for i,(x,s) in enumerate([(9*PITCH,1),(-9*PITCH,1),(9*PITCH,-1),(-9*PITCH,-1)]):
        y=s*9.5*PITCH
        motor=part(chassis,f'motor_{i}','V5 Smart Motor envelope',(x,s*.061,0))
        element(motor,'geom',name=f'motor_{i}_case',type='box',size='.025 .030 .024',mass=.280,rgba='.76 .1 .12 1')
        cosmetic(motor,f'motor_{i}_cap',pos=f'0 {-s*.024} 0',size='.025 .007 .024',rgba=DARK)
        # The inner bearing's two screws also engage the motor inserts.
        connections.append(dict(part=f'motor_{i}',to=f'{"left" if s==1 else "right"}_inner_rail',type='bolted through bearing',fasteners=[f'bearing_{i}_inner_bolt_0',f'bearing_{i}_inner_bolt_1']))
        for support,offset in [('inner',7.5*PITCH),('outer',11.5*PITCH)]:
            by=s*(offset+(.003 if support=='outer' else -.003))
            bearing=part(chassis,f'bearing_{i}_{support}','bearing flat',(x,by,0))
            element(bearing,'geom',name=f'bearing_{i}_{support}_plate',type='box',size='.0185 .0022 .006',mass=.004,contype=0,conaffinity=0,rgba=DARK)
            for j,dx in enumerate([-PITCH,PITCH]):
                bolt(chassis,f'bearing_{i}_{support}_bolt_{j}',(x+dx,by,0),axis='y',nut=support=='outer')
            connections.append(dict(part=f'bearing_{i}_{support}',to=f'{"left" if s==1 else "right"}_{support}_rail',type='bolted',fasteners=2))
        rotating=part(chassis,f'axle_wheel_{i}','shaft and wheel rotating assembly',(x,y,0))
        element(rotating,'joint',name=f'axle_{i}',type='hinge',axis='0 1 0',damping=.001,armature=.0001)
        element(rotating,'geom',name=f'square_axle_{i}',type='box',pos=f'0 {-s*.009} 0',size='.0015875 .04445 .0015875',mass=.007,contype=0,conaffinity=0,rgba='.75 .78 .81 1')
        wheel=part(rotating,f'wheel_{i}','4 inch traction wheel')
        element(wheel,'geom',name=f'tire_{i}',type='cylinder',size=f'{RADIUS} .0127',quat='.707106781 .707106781 0 0',mass=.120,rgba='.035 .04 .048 1')
        for outward in (-1,1):
            cosmetic(wheel,f'hub_{i}_{outward}',type='cylinder',pos=f'0 {outward*.0129} 0',size='.028 .0008',quat='.707106781 .707106781 0 0',rgba='.56 .59 .63 1')
            for k in range(6):
                a=k*math.pi/3
                cosmetic(wheel,f'hub_recess_{i}_{outward}_{k}',type='cylinder',pos=vec([.018*math.cos(a),outward*.0138,.018*math.sin(a)]),size='.006 .0003',quat='.707106781 .707106781 0 0',rgba=DARK)
        # Collars capture the wheel axially; bearing/shaft radial constraint is the hinge.
        for j,offset in enumerate([-.018,.018]):
            collar=part(rotating,f'collar_{i}_{j}','shaft collar',(0,offset,0))
            element(collar,'geom',name=f'collar_{i}_{j}_ring',type='cylinder',size='.0055 .003',quat='.707106781 .707106781 0 0',mass=.003,contype=0,conaffinity=0,rgba=SILVER)
            cosmetic(collar,f'collar_{i}_{j}_set_screw',pos='.005 0 0',size='.0015 .002 .002',rgba=DARK)
        connections.append(dict(part=f'axle_wheel_{i}',to=[f'bearing_{i}_inner',f'bearing_{i}_outer',f'motor_{i}'],type='revolute shaft support and direct drive'))
    actuator=element(root,'actuator')
    for i in range(4):
        element(actuator,'motor',name=f'drive_{i}',joint=f'axle_{i}',ctrllimited='true',ctrlrange='-1.05 1.05')
    catalog=dict(description='Fixed V5-style standard four-wheel chassis assembly test',
        units='meters, kilograms, seconds; +x forward, +y left, +z up',
        nominal_geometry=dict(hole_pitch_m=PITCH,channel_length_m=25*PITCH,wheel_diameter_m=2*RADIUS,
                              wheelbase_m=18*PITCH,track_width_m=19*PITCH,shaft_square_width_m=.003175),
        parts=parts,connections=connections,sources=SOURCES,
        limitations=['Nominal VEX-style parts, not imported manufacturer CAD or a certified build.',
                     'C-channel perforations are visual meshes; simple hidden collision proxies and approximate component masses are used.',
                     'Bolted/strapped connections are rigid child bodies, not simulated screw thread engagement.',
                     'Shaft rotation is constrained by a hinge; bearing hole friction and fastener stress are not simulated.',
                     'Motor/brain/battery case dimensions and fastening details are simplified.'])
    return ET.tostring(root,encoding='unicode'),catalog


def settle(model,seconds=3):
    data=mujoco.MjData(model)
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model,data)
    touching=set()
    for c in data.contact:
        if c.dist>.0001: continue
        names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c.geom1,c.geom2)}
        if 'floor' in names:
            touching.update(n for n in names if n and n.startswith('tire_'))
    tilt=math.acos(float(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1)))
    result=dict(standing=bool(len(touching)==4 and tilt<.01 and np.isfinite(data.qpos).all() and not np.any(data.warning.number)),
                simulated_seconds=float(data.time),wheels_touching_floor=sorted(touching),tilt_degrees=math.degrees(tilt),
                root_height_m=float(data.qpos[2]),root_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
                mass_kg=float(sum(model.body_mass)),degrees_of_freedom=model.nv,
                fixed_body_count=model.nbody-6,root_pose_driven=False)
    return data,result


def render(directory):
    from PIL import Image
    directory=Path(directory)
    model=mujoco.MjModel.from_xml_path(str((directory/'chassis.xml').resolve()))
    data,_=settle(model)
    with mujoco.Renderer(model,height=1000,width=1400) as renderer:
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.055];camera.distance=.83
        for name,azimuth,elevation in [('chassis',135,-32),('top',90,-89),('axles',90,-13)]:
            camera.azimuth=azimuth;camera.elevation=elevation
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(directory/f'{name}.png')


def run(directory=OUTPUT,viewer=False,render_images=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    xml,catalog=build();(directory/'chassis.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml)
    data,result=settle(model)
    (directory/'assembly.json').write_text(json.dumps(catalog,indent=2)+'\n')
    if render_images:
        try:
            p=subprocess.run([sys.executable,'-m','chassis_lab.assembly','render',str(directory)],capture_output=True,timeout=30)
            result['rendered']=p.returncode==0
            if p.returncode: result['render_error']='Graphics unavailable; use macOS desktop graphics access or --no-render.'
        except Exception as exc:
            result['rendered']=False;result['render_error']=type(exc).__name__
    (directory/'standing_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Assembly:',directory.resolve())
    if viewer:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model,data) as window:
            window.cam.lookat[:]=[0,0,.06];window.cam.distance=.8;window.cam.azimuth=135;window.cam.elevation=-30
            while window.is_running():
                started=time.monotonic()
                for _ in range(10):mujoco.mj_step(model,data)
                window.sync();time.sleep(max(0,.02-(time.monotonic()-started)))
    return result['standing']


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='render':render(sys.argv[2])
