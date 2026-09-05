"""CAD assembly visualization and explicitly separate contact simulation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .physics import element

ROOT=Path(__file__).resolve().parents[1]


def tree(name):
    return json.loads((ROOT/'data'/'cad_assets'/name/'assembly_tree.json').read_text())


def instance_pose(item,world=None):
    pose=np.array(item['transform_mm'],dtype=float);pose[:3,3]/=1000
    return pose if world is None else world@pose


def add_visual(asset,parent,item,definition,prefix,world=None):
    mesh_id=prefix+'_'+definition['id']
    if asset.find(f"mesh[@name='{mesh_id}']") is None:
        element(asset,'mesh',name=mesh_id,file=str(ROOT/definition['mesh']),scale='.001 .001 .001',inertia='shell')
    pose=instance_pose(item,world);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,pose[:3,:3].ravel())
    element(parent,'geom',name=prefix+'_'+item['id'],type='mesh',mesh=mesh_id,
            pos=' '.join(map(str,pose[:3,3])),quat=' '.join(map(str,quat)),
            contype=0,conaffinity=0,mass=0,group=1,rgba=' '.join(map(str,definition['rgba'])))


def scene_base():
    root=ET.Element('mujoco',model='cad_field_and_clawbot')
    element(root,'compiler',angle='radian',autolimits='true')
    element(root,'option',gravity='0 0 -9.81',timestep='.002',integrator='implicitfast',iterations=80)
    visual=element(root,'visual');element(visual,'global',offwidth=1200,offheight=900)
    element(visual,'headlight',ambient='.4 .4 .4',diffuse='.65 .65 .65')
    asset=element(root,'asset');world=element(root,'worldbody')
    element(world,'light',pos='0 0 5',dir='0 0 -1')
    return root,asset,world


def inspection(name):
    data=tree(name);root,asset,world=scene_base()
    lo,hi=np.array(data['bounds_mm'])/1000;size=hi-lo
    # Field exports may be Y-up; inspect all three spans rather than assume Z.
    rot=np.eye(3)
    if name=='override_field' and np.argmin(size)==1:rot=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@rot.T
    lower=corners.min(axis=0);upper=corners.max(axis=0)
    transform=np.eye(4);transform[:3,:3]=rot;transform[:3,3]=[-(lower[0]+upper[0])/2,-(lower[1]+upper[1])/2,-lower[2]]
    for item in data['instances']:add_visual(asset,world,item,data['definitions'][item['definition']],name,transform)
    element(world,'geom',name='inspection_floor',type='plane',size='6 6 .1',rgba='.18 .21 .24 1')
    out=ROOT/'data'/'cad_assets'/name/'assembly_inspection.xml';out.write_text(ET.tostring(root,encoding='unicode'))
    (out.parent/'scene_transform.json').write_text(json.dumps(dict(transform_m=transform.tolist(),size_m=(upper-lower).tolist()),indent=2))
    return out


def render(path,label='overview',close=False,qpos=None):
    from PIL import Image
    path=Path(path);m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    if qpos is not None:d.qpos[:]=qpos
    mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.1];cam.distance=6.2;cam.azimuth=125;cam.elevation=-60
    if close:cam.lookat[:]=[0,0,.13];cam.distance=1.05;cam.elevation=-25
    with mujoco.Renderer(m,height=900,width=1200) as r:
        r.update_scene(d,camera=cam);output=path.with_name(label+'.png');Image.fromarray(r.render()).save(output)
    print(output,flush=True)
    return output


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('asset',choices=['clawbot','override_field']);a=p.parse_args()
    render(inspection(a.asset),close=a.asset=='clawbot')
