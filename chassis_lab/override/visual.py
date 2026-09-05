import json
from pathlib import Path
import time
import mujoco
import numpy as np


def load(path):
    path=Path(path)
    m=mujoco.MjModel.from_xml_path(str(path.with_suffix('.xml').resolve()));d=mujoco.MjData(m)
    t=np.load(path.with_suffix('.npz')) if path.with_suffix('.npz').exists() else None
    return m,d,t


def render(path):
    from PIL import Image
    path=Path(path);m,d,t=load(path)
    c=mujoco.MjvCamera();c.distance=1.35;c.azimuth=125;c.elevation=-30;c.lookat[:]=[.4,0,.14]
    if 'full_field' in path.name:c.distance=6;c.lookat[:]=[1.783,1.783,.1];c.elevation=-65
    with mujoco.Renderer(m,height=700,width=1000) as renderer:
        indices=[0,len(t['qpos'])//2,-1] if t is not None else [0]
        for label,i in zip(['start','middle','finish'],indices):
            if t is not None:d.qpos[:]=t['qpos'][i]
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera=c)
            Image.fromarray(renderer.render()).save(path.with_name(path.stem+'_'+label+'.png'))
        if t is not None:
            frames=[]
            for i in np.linspace(0,len(t['qpos'])-1,32).astype(int):
                d.qpos[:]=t['qpos'][i];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=c)
                frames.append(Image.fromarray(renderer.render()).resize((600,420)))
            frames[0].save(path.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=130,loop=0)


def replay(path,headless=False):
    m,d,t=load(path)
    if headless:
        for q in t['qpos'] if t is not None else [d.qpos.copy()]:d.qpos[:]=q;mujoco.mj_forward(m,d)
        print(json.dumps(dict(finite=bool(np.isfinite(d.qpos).all()),frames=len(t['qpos']) if t is not None else 1)));return
    from mujoco import viewer
    with viewer.launch_passive(m,d) as v:
        v.cam.lookat[:]=[.4,0,.15];v.cam.distance=1.4;v.cam.azimuth=125;v.cam.elevation=-30
        if 'full_field' in str(path):v.cam.lookat[:]=[1.783,1.783,.1];v.cam.distance=6;v.cam.elevation=-65
        if 'assembly_inspection' in str(path) and 'override_field' in str(path):v.cam.lookat[:]=[0,0,.1];v.cam.distance=6;v.cam.elevation=-65
        i=0
        while v.is_running():
            start=time.monotonic()
            if t is not None:
                d.qpos[:]=t['qpos'][i];mujoco.mj_forward(m,d);i=(i+1)%len(t['qpos'])
                if 'cad_drive' in str(path):v.cam.lookat[:]=d.qpos[:3]+[0,0,.09]
            else:
                for _ in range(25):mujoco.mj_step(m,d)
            v.sync();time.sleep(max(0,.05-(time.monotonic()-start)))


if __name__=='__main__':
    import sys
    render(sys.argv[1])
