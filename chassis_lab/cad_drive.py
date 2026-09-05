"""Imported CAD driving experiment; contact proxies and dynamics are assumptions."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import importlib.metadata
import mujoco
import numpy as np
from .cad_scene import ROOT, tree, scene_base, add_visual, instance_pose
from .physics import element


def vec(values):
    return ' '.join(str(float(v)) for v in values)


def compile_scene(output,lightweight=True,crop=True,robot_data=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root,asset,world=scene_base()
    default=element(root,'default')
    element(default,'geom',friction='.7 .002 .0001',solref='.01 1',solimp='.95 .99 .001')
    field=tree('override_field');robot=robot_data if robot_data is not None else tree('clawbot')
    # Export is Y-up. The upper surface of the 36 floor tiles defines z=0.
    ft=np.eye(4);ft[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    tiles=[i for i in field['instances'] if i['definition']=='p00032']
    ft[2,3]=-max(instance_pose(i)[1,3] for i in tiles)
    element(world,'geom',name='floor',type='plane',size='4 4 .1',group=3,rgba='.16 .18 .20 1')
    element(world,'geom',name='display_ground',type='plane',pos='0 0 -.022',size='4 4 .1',contype=0,conaffinity=0,rgba='.16 .18 .20 1')
    proxies=[]
    for item in field['instances']:
        definition=field['definitions'][item['definition']]
        add_visual(asset,world,item,definition,'field',ft)
        geom=world[-1];name=definition['name']
        if item['definition']=='p00032':geom.set('rgba','.32 .34 .36 1')
        elif 'Bright Red' in name or 'Red Loader' in name:geom.set('rgba','.65 .035 .055 1')
        elif 'Bright Blue' in name or 'Blue Loader' in name:geom.set('rgba','.025 .3 .72 1')
        elif 'Yellow' in name:geom.set('rgba','.9 .68 .035 1')
        elif 'Tape' in name:geom.set('rgba','.9 .9 .9 1')
        elif 'Clear' in name:geom.set('rgba','.65 .8 .88 .45')
        else:geom.set('rgba','.46 .48 .52 1')
        # Fixed convex part proxies: no expensive triangle contacts; openings
        # in cups/goals are deliberately NOT a calibrated manipulation model.
        span=np.diff(np.array(definition['bounds_mm']),axis=0)[0]/1000
        if item['definition']=='p00032' or 'Tape' in name or 'Wall Alliance' in name or max(span)<.04:
            continue
        pose=instance_pose(item,ft)
        if max(abs(pose[0,3]),abs(pose[1,3]))>1.9:continue
        attrs={k:geom.attrib[k] for k in ['mesh','pos','quat']}
        element(world,'geom',name='contact_'+item['id'],type='mesh',**attrs,
                contype=1,conaffinity=2,group=3,rgba='0 .7 0 .15',mass=0)
        proxies.append(item['id'])
    body=element(world,'body',name='robot',pos='-.9 -.95 .056')
    element(body,'freejoint',name='root')
    # CAD has no trustworthy material/mass properties. Explicit approximate
    # masses cover fixed frame/electronics, battery and fixed arm separately.
    element(body,'geom',name='frame_contact',type='box',pos='0 0 .018',size='.126 .114 .012',mass=1.35,group=3,contype=2,conaffinity=1,rgba='0 .7 0 .2')
    element(body,'geom',name='battery_mass',type='box',pos='-.1 -.014 .01',size='.024 .08 .015',mass=.35,group=3,contype=0,conaffinity=0,rgba='0 0 0 0')
    element(body,'geom',name='fixed_arm_mass',type='box',pos='.06 0 .115',size='.13 .025 .018',mass=.65,group=3,contype=0,conaffinity=0,rgba='0 0 0 0')
    element(body,'geom',name='fixed_claw_contact',type='box',pos='.25 .005 .042',size='.082 .06 .016',mass=0,group=3,contype=2,conaffinity=1,rgba='0 .7 0 .2')
    element(body,'geom',name='arm_contact',type='capsule',fromto='-.122 0 .149 .16 0 .06',size='.018',mass=0,group=3,contype=2,conaffinity=1,rgba='0 .7 0 .2')
    element(body,'geom',name='mast_contact',type='box',pos='-.122 0 .105',size='.018 .045 .095',mass=0,group=3,contype=2,conaffinity=1,rgba='0 .7 0 .2')
    rt=np.eye(4);rt[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    wheel_items=[i for i in robot['instances'] if i['definition'] in ['p00010','p00012']]
    def wheel_center(item,transform=None):
        pose=instance_pose(item,transform)
        if robot_data is not None:
            midpoint=np.mean(robot['definitions'][item['definition']]['bounds_mm'],axis=0)[1]/1000
            return pose[:3,3]+pose[:3,1]*midpoint
        return pose[:3,3]
    centers=np.array([wheel_center(i) for i in wheel_items])
    rt[:3,3]=-rt[:3,:3]@centers.mean(axis=0)
    actuator=element(root,'actuator');wheels=[]
    for item in robot['instances']:
        definition=robot['definitions'][item['definition']]
        if item not in wheel_items:
            add_visual(asset,body,item,definition,'robot',rt);continue
        center=wheel_center(item,rt);name='wheel_'+item['id']
        wb=element(body,'body',name=name,pos=vec(center))
        element(wb,'joint',name=name,type='hinge',axis='0 1 0',damping='.001',armature='.00002')
        local=rt.copy();local[:3,3]-=center
        add_visual(asset,wb,item,definition,'robot',local)
        driven=item['id'] in {'i00010','i00024'}
        traction=item['definition']=='p00010';radius=.0508 if traction else .05305
        halfwidth=(definition['bounds_mm'][1][1]-definition['bounds_mm'][0][1])/2000 if robot_data is not None else .0095
        element(wb,'geom',name=name+'_contact',type='cylinder',size=f'{radius} {halfwidth}',quat='.7071067812 .7071067812 0 0',
                mass=.12,friction='.8 .001 .00005' if traction else '.04 .001 .00005',priority=1,group=3,contype=2,conaffinity=1,rgba='0 .7 0 .2')
        if driven:element(actuator,'motor',name=name,joint=name,ctrllimited='true',ctrlrange='-.5 .5')
        wheels.append(dict(name=name,center_m=center.tolist(),radius_m=radius,driven=driven,source_instance=item['id']))
    from .cad_lightweight import optimize
    optimization=optimize(root,ft,lightweight,crop)
    path=output/'cad_drive.xml';path.write_text(ET.tostring(root,encoding='unicode'))
    config=dict(version='cad-drive-1',source_hashes={n:tree(n)['source_sha256'] for n in ['clawbot','override_field']},
        wheels=wheels,field_contact_proxies=len(proxies),field_transform=ft.tolist(),robot_transform=rt.tolist(),
        assumptions=['CAD is visual geometry; collision proxies use convex hulls of larger field parts.',
        'Field game objects and Toggles are fixed in this driving experiment; goal/cup openings are not usable for scoring.',
        'Arm and claw are locked at their supplied pose. Fasteners are rigid attachments; STEP supplies no verified mates.',
        'Two driven axle positions, two passive axle positions. Wheel type follows the assembly specification; omni lateral behavior is approximated with low isotropic friction, not individual rollers.',
        'Masses: frame/electronics 1.35 kg, battery .35 kg, fixed arm/claw .65 kg, each wheel .12 kg. These are estimates, not verified CAD mass properties.',
        'Motor approximation: .5 Nm stall, 200 rpm no-load, linear torque-speed envelope; model is not a calibrated V5 motor.',
        'No AI proposals or API calls in this CAD import/driving demonstration.'])
    config['dependencies']={p:importlib.metadata.version(p) for p in ['mujoco','numpy']}
    config['timestep_s']=.002
    config['optimization']=optimization
    if robot_data is not None:
        config['variant_name']=robot_data.get('variant_name')
        config['assumptions'][-1]='Source-part variant selection is recorded separately in the parent design and API records; this compiler does not call the API.'
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    return path,config


def simulate(path,config,duration=10):
    m=mujoco.MjModel.from_xml_path(str(Path(path).resolve()));d=mujoco.MjData(m)
    wheels=config['wheels'];driven=[w for w in wheels if w['driven']]
    ids=[m.joint(w['name']).id for w in driven]
    qpos=[];times=[];episodes=0;saturation=0;max_tilt=0;last_contact=-math.inf
    settled=None;straight=None;turned=None
    wheel_addresses=[m.jnt_qposadr[m.joint(w['name']).id] for w in wheels]
    travel=np.zeros(len(wheels));previous=d.qpos[wheel_addresses].copy()
    for step in range(round(duration/m.opt.timestep)):
        t=d.time
        # Fixed time program: settle, accelerate, brake, turn, brake, drive.
        phase='settle' if t<1 else 'straight' if t<4 else 'brake' if t<5 else 'turn' if t<7 else 'brake' if t<8 else 'straight'
        for j,w in zip(ids,driven):
            target=0 if phase in ['settle','brake'] else 5.0
            if phase=='turn':target=(-3 if w['center_m'][1]>0 else 3)
            speed=d.qvel[m.jnt_dofadr[j]];requested=.1*(target-speed)
            # Motoring falls linearly to zero; braking bounded by stall torque.
            limit=.5*max(0,1-abs(speed)/(200*2*math.pi/60)) if requested*speed>0 else .5
            a=m.actuator(w['name']).id;d.ctrl[a]=np.clip(requested,-limit,limit)
            saturation+=int(abs(requested)>limit)
        mujoco.mj_step(m,d)
        travel+=abs(d.qpos[wheel_addresses]-previous);previous=d.qpos[wheel_addresses].copy()
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite dynamics')
        tilt=math.acos(np.clip(d.xmat[m.body('robot').id,8],-1,1));max_tilt=max(max_tilt,tilt)
        active=set()
        for c in d.contact:
            a,b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)
            if a.startswith('contact_') or b.startswith('contact_'):active.add(tuple(sorted([a,b])))
        if active:
            if t-last_contact>.2:episodes+=1
            last_contact=t
        if t>=1 and settled is None:settled=d.qpos[:7].copy()
        if t>=4 and straight is None:straight=d.qpos[:7].copy()
        if t>=7 and turned is None:turned=d.qpos[:7].copy()
        if step%25==0:qpos.append(d.qpos.copy());times.append(d.time)
    def yaw(q):return math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2))
    metrics=dict(duration_s=d.time,total_mass_kg=float(m.body_mass.sum()),finite=True,
        straight_displacement_m=float(np.linalg.norm(straight[:2]-settled[:2])),
        turn_heading_change_deg=float(math.degrees(math.atan2(math.sin(yaw(turned)-yaw(straight)),math.cos(yaw(turned)-yaw(straight))))),
        total_displacement_m=float(np.linalg.norm(d.qpos[:2]-settled[:2])),
        max_tilt_deg=math.degrees(max_tilt),obstacle_contact_episodes=episodes,
        collision_episode_clearance_s=.2,
        saturation_fraction=saturation/(round(duration/m.opt.timestep)*len(driven)),
        wheel_travel_m={w['name']:float(travel[k]*w['radius_m']) for k,w in enumerate(wheels)},
        final_position_m=d.qpos[:3].tolist(),root_forces_used=False,protocol='fixed drive/brake/turn; not an Override scoring benchmark')
    np.savez_compressed(Path(path).with_suffix('.npz'),qpos=qpos,time=times)
    Path(path).with_name('metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2));return metrics


def render_run(path):
    from PIL import Image
    path=Path(path);m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m)
    trajectory=np.load(path.with_suffix('.npz'))['qpos'];cam=mujoco.MjvCamera()
    cam.azimuth=125;cam.elevation=-35;cam.distance=1.45
    with mujoco.Renderer(m,height=600,width=800) as r:
        frames=[]
        for index in np.linspace(0,len(trajectory)-1,50).astype(int):
            d.qpos[:]=trajectory[index];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[0,0,.09]
            r.update_scene(d,camera=cam);frames.append(Image.fromarray(r.render()).resize((800,600)))
        frames[0].save(path.with_name('start.png'));frames[-1].save(path.with_name('finish.png'))
        frames[0].save(path.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=200,loop=0)
        config=json.loads(path.with_name('configuration.json').read_text())
        local=config.get('optimization',{}).get('local_field',False)
        cam.lookat[:]=[-.475,-.9,.05] if local else [0,0,.05];cam.distance=3 if local else 6.2;cam.elevation=-65
        r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(path.with_name('field.png'))


def run(output='assemblies/cad_drive',render=True,lightweight=True,crop=True):
    path,config=compile_scene(output,lightweight,crop);metrics=simulate(path,config)
    if render:
        try:render_run(path)
        except Exception as exc:Path(output,'render_error.txt').write_text(str(exc))
    import html
    Path(output,'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Imported CAD driving test</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#172029;color:#eee}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Imported CAD driving test</h1><p>Physical wheel actuation; handwritten test program. Arm, claw and field objects fixed.</p><img src="cad_drive.gif"><h2>Measured results</h2><pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre><h2>Model limitations</h2><pre>'+html.escape('\n'.join(config['assumptions']))+'</pre><img src="field.png">')
    return path
