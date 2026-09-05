"""Isolated rendering/replay; failures never change physics metrics."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco


def load_trial(directory, split, seed):
    directory=Path(directory)
    xml=directory/f'{split}_{seed}.xml'
    model=mujoco.MjModel.from_xml_path(str(xml.resolve()))
    data=mujoco.MjData(model)
    trajectory=np.load(directory/f'{split}_{seed}.npz')
    return model,data,trajectory


def camera():
    c=mujoco.MjvCamera()
    c.lookat[:]=[2.8,0,0]
    c.distance=8.0; c.azimuth=90; c.elevation=-60
    return c


def snapshot(directory, split, seed):
    from PIL import Image
    model,data,trajectory=load_trial(directory,split,seed)
    with mujoco.Renderer(model,height=700,width=1000) as renderer:
        for label,index in [('start',0),('slalom',len(trajectory['qpos'])//2),('finish',-1)]:
            data.qpos[:]=trajectory['qpos'][index]; mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera())
            Image.fromarray(renderer.render()).save(Path(directory)/f'{label}.png')
        c=camera(); c.lookat[:]=data.qpos[:3]; c.distance=0.9; c.azimuth=135; c.elevation=-30
        renderer.update_scene(data,camera=c)
        Image.fromarray(renderer.render()).save(Path(directory)/'robot.png')


def replay(directory, candidate=0, split='final', seed=101, headless=False):
    directory=Path(directory)/f'candidate_{candidate}'
    model,data,trajectory=load_trial(directory,split,seed)
    if headless:
        for q in trajectory['qpos']:
            data.qpos[:]=q; mujoco.mj_forward(model,data)
        print(json.dumps({'replay_frames':len(trajectory['qpos']),'finite':bool(np.isfinite(data.qpos).all()),'note':'Recorded-state playback, not a new physics evaluation'}))
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=camera().lookat
        viewer.cam.distance=8; viewer.cam.azimuth=90; viewer.cam.elevation=-60
        for i,q in enumerate(trajectory['qpos']):
            if not viewer.is_running(): break
            start=time.monotonic()
            data.qpos[:]=q; mujoco.mj_forward(model,data); viewer.sync()
            delay=trajectory['time'][i+1]-trajectory['time'][i] if i+1<len(trajectory['time']) else .02
            time.sleep(max(0,float(delay)-(time.monotonic()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('split'); p.add_argument('seed',type=int)
    args=p.parse_args()
    snapshot(args.directory,args.split,args.seed)
