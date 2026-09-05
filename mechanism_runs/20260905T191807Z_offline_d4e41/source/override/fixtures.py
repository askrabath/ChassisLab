import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ..physics import element
from . import field


def base():
    root=ET.Element('mujoco',model='override_contact_fixture')
    element(root,'compiler',angle='radian')
    element(root,'option',timestep='.002',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic',iterations=80)
    default=element(root,'default');element(default,'geom',friction='.65 .002 .0001',solref='.01 1')
    visual=element(root,'visual');element(visual,'global',offwidth=1200,offheight=900)
    element(visual,'headlight',ambient='.5 .5 .5',diffuse='.7 .7 .7')
    element(root,'asset');element(root,'worldbody')
    asset,world=field.environment(root)
    element(world,'light',pos='0 0 2',dir='0 0 -1')
    return root,asset,world


def validate(directory='assemblies/override_fixtures'):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);results={}
    for name in ['pin_goal','pin_cup_stack','outside_goal','falling_cup','full_field']:
        root,a,w=base()
        if name=='full_field':field.environment(root,full=True)
        else:
            field.goal(a,w,'goal',(0,0,0))
            if name=='falling_cup':field.cup(a,w,'cup',(.2,0,.3))
            else:
                field.pin(a,w,'pin',(.11 if name=='outside_goal' else 0,0,.12))
                if name=='pin_cup_stack':
                    field.cup(a,w,'cup',(0,0,.20));field.pin(a,w,'upper_pin',(0,0,.31),('yellow','red'))
        xml=ET.tostring(root,encoding='unicode');(out/f'{name}.xml').write_text(xml)
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);q=[];times=[]
        for i in range(2000):
            mujoco.mj_step(m,d)
            if i%25==0:q.append(d.qpos.copy());times.append(float(d.time))
        item=dict(finite=bool(np.isfinite(d.qpos).all()),warnings=int(sum(d.warning.number)),max_speed=float(np.max(np.abs(d.qvel))) if m.nv else 0)
        if name in ('pin_goal','pin_cup_stack','outside_goal'):
            bid=m.body('pin').id
            item['pin_placed']=field.nesting(d.xpos[bid],d.xmat[bid].reshape(3,3)[:,2],np.zeros(2),field.GOAL_H,.01)
            item['pin_z']=float(d.xpos[bid,2])
        if name=='pin_cup_stack':
            item['cup_z']=float(d.xpos[m.body('cup').id,2]);item['upper_pin_z']=float(d.xpos[m.body('upper_pin').id,2])
            item['stable_stack']=bool(item['pin_placed'] and item['cup_z']>.14 and item['upper_pin_z']>.25 and item['max_speed']<.02)
        if name=='full_field':item['toggles']=4
        results[name]=item;np.savez_compressed(out/f'{name}.npz',qpos=np.array(q),time=np.array(times))
    results['scoring_negative_tests']={
        'above_rim_rejected':not field.nesting([0,0,.30],[0,0,1],[0,0],field.GOAL_H,.01),
        'near_but_outside_rejected':not field.nesting([.07,0,.1],[0,0,1],[0,0],field.GOAL_H,.01),
        'horizontal_two_half_rejected':not field.nesting([0,0,.05],[1,0,0],[0,0],field.GOAL_H,.01)}
    (out/'validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    return results

if __name__=='__main__':validate()
