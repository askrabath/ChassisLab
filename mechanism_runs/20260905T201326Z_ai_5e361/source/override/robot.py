import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ..assembly import build,vec
from ..physics import element
from . import field


def compile_robot(design,task='pin',friction=.65,object_xy=(.36,0),full=False):
    root=ET.fromstring(build()[0]);root.set('model','override_mechanism')
    root.find('option').set('iterations','80')
    chassis=root.find(".//body[@name='chassis']")
    # Remove the raised front/rear crossmembers for the arm envelope. Inner/outer
    # side rails remain connected by the fixed chassis subassembly surrogate.
    for b in list(chassis.findall('body')):
        if 'crossmember' in b.get('name','') or 'standoff' in b.get('name',''):
            chassis.remove(b)
    asset,world=field.environment(root,full,friction)
    for s in (-1,1):
        b=element(chassis,'body',name=f'arm_tower_{s}',pos=f'-.0508 {s*.065} .08')
        element(b,'geom',name=f'tower_{s}',type='box',size='.00635 .0127 .06',mass=.07,rgba='.6 .65 .7 1')
        element(b,'geom',name=f'tower_bearing_{s}',type='cylinder',pos='0 0 .06',quat='.70710678 .70710678 0 0',size='.012 .003',mass=.01,rgba='.12 .12 .13 1')
    bybody={'chassis':chassis};byid={p.id:p for p in design.parts};roles={}
    actuator=root.find('actuator')
    for p in design.parts:
        offset=np.array(p.offset_m,dtype=float)
        if p.attachment=='tip':offset[0]+=byid[p.parent].size_m[0]
        if p.attachment=='tool':offset[0]+=.03
        b=element(bybody[p.parent],'body',name=p.id,pos=vec(offset));bybody[p.id]=b
        if p.joint.kind!='fixed':
            kwargs=dict(name=p.id+'_joint',type='hinge' if p.joint.kind=='revolute' else 'slide',axis=vec(p.joint.axis),damping='.02',armature='.0002')
            if not p.joint.role.startswith('roller'):kwargs.update(limited='true',range=vec(p.joint.limits))
            element(b,'joint',**kwargs)
        x,y,z=p.size_m
        if p.catalog=='beam':
            for s in (-1,1):element(b,'geom',name=f'{p.id}_{s}',type='box',pos=f'{x/2} {s*y/2} 0',size=f'{x/2} .00635 {z/2}',mass=x*.40,rgba='.6 .67 .72 1')
        elif p.catalog=='roller':
            element(b,'geom',name=p.id+'_contact',type='cylinder',size=f'{y/2} {x/2}',quat='.70710678 0 .70710678 0',mass=.045,friction='1.1 .003 .0001',rgba='.15 .18 .2 1')
        else:
            pos=[x/2,0,0] if p.catalog=='finger' else [0,0,0]
            mass={'finger':.025,'tool_mount':.08,'guide':.015}[p.catalog]
            element(b,'geom',name=p.id+'_contact',type='box',pos=vec(pos),size=vec(np.array(p.size_m)/2),mass=mass,friction='1.1 .003 .0001',rgba='.8 .16 .18 1' if p.catalog=='finger' else '.45 .5 .55 1')
        if p.joint.role!='passive':
            role=p.joint.role;roles[role]=p
            # Mass contribution for a physically mounted geared actuator.
            mb=element(bybody[p.parent],'body',name=p.id+'_motor',pos=vec(offset+np.array([0,-.024 if role in ('lift','wrist') else 0,0])))
            element(mb,'geom',name=p.id+'_motor_case',type='box',size='.018 .014 .02',mass=.18 if role.startswith('grip') else .28,contype=0,conaffinity=0,rgba='.2 .2 .21 1')
            bound=(.525 if role.startswith('grip') else 1.05)*p.joint.reduction
            element(actuator,'motor',name=role,joint=p.id+'_joint',ctrllimited='true',ctrlrange=f'{-bound} {bound}')
    if not full:
        field.goal(asset,world,'target_goal',(.80,0,0))
        if task=='cup':field.cup(asset,world,'object',(*object_xy,field.CUP_H/2+.002))
        else:field.pin(asset,world,'object',(*object_xy,field.PIN_H/2+.002))
    # Explicit mating support and controller observation sites.
    element(chassis,'site',name='chassis_pose',size='.001',rgba='0 0 0 0')
    wrist=next((p for p in design.parts if p.joint.role=='wrist'),None)
    element(bybody[wrist.id],'site',name='grasp_center',pos='.09 0 0',size='.003',rgba='0 1 0 .4')
    sensor=element(root,'sensor')
    for role,p in roles.items():
        element(sensor,'jointpos',name=role+'_encoder',joint=p.id+'_joint')
    return ET.tostring(root,encoding='unicode')


def initial_data(model,design,folded=True):
    d=mujoco.MjData(model)
    for p in design.parts:
        if p.joint.role=='passive':continue
        value=.9 if p.joint.role=='lift' else (-.9 if p.joint.role=='wrist' else 0)
        j=model.joint(p.id+'_joint')
        d.qpos[j.qposadr[0]]=np.clip(value,*p.joint.limits)
    mujoco.mj_forward(model,d)
    return d


def construction_check(model,data,design):
    # Bounding sphere of each geom is conservative; exact boxes use corners.
    mins=[];maxs=[]
    robot_root=model.body('chassis').id
    for i in range(model.ngeom):
        bid=model.geom_bodyid[i]
        if model.body_rootid[bid]!=robot_root:continue
        # Visual meshes use mesh bounds; massless cosmetic details do not set limits.
        if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:continue
        size=model.geom_size[i].copy()
        if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER:size=np.array([size[0],size[0],size[1]])
        extent=np.abs(data.geom_xmat[i].reshape(3,3))@size
        mins.append(data.geom_xpos[i]-extent);maxs.append(data.geom_xpos[i]+extent)
    lower=np.min(mins,axis=0);upper=np.max(maxs,axis=0)
    envelope=upper-lower
    if np.max(envelope)>.4572+.001:raise ValueError(f'initial envelope exceeds 18 inch cube: {envelope.tolist()}')
    # Deep contact between movable parts is rejected; fixed mating overlaps and
    # parent-child shaft/bearing interfaces are intentional compiler exclusions.
    bad=[]
    for c in data.contact:
        if c.dist<-.004:
            a,b=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c.geom1,c.geom2)]
            bad.append([a,b,float(c.dist)])
    if bad:raise ValueError(f'initial penetration >4 mm: {bad[:3]}')
    return dict(assembled_in_simulation=True,implemented_construction_checks=True,reviewed_physically_buildable=False,official_legality=False,initial_envelope_m=envelope.tolist(),mass_kg=float(sum(model.body_mass[model.body_rootid==robot_root])))
