"""Repeatable drivetrain and friction sensitivity checks; no API calls."""
import json
from pathlib import Path
import mujoco
import numpy as np
from chassis_lab.override.design import seed,apply_edits,fixture_edit
from chassis_lab.override.robot import compile_robot,initial_data
from chassis_lab.override.sim import Motors,run
from chassis_lab.physics import yaw_of

out=Path('assemblies/mechanism_verification');out.mkdir(parents=True,exist_ok=True)
robot=seed();results={}
for name,v,w in [('forward',.25,0),('turn',0,1)]:
    m=mujoco.MjModel.from_xml_string(compile_robot(robot));d=initial_data(m,robot);control=Motors(m,robot)
    for _ in range(1000):control.apply(d,v,w,.9,0);mujoco.mj_step(m,d)
    driven=dict(displacement_m=float(np.linalg.norm(d.qpos[:2])),yaw_rad=float(yaw_of(d.qpos[3:7])))
    for _ in range(1500):control.apply(d,0,0,.9,0);mujoco.mj_step(m,d)
    driven.update(braked_speed_m_s=float(np.linalg.norm(d.qvel[:2])),braked_yaw_rate_rad_s=float(abs(d.qvel[5])),finite=bool(np.isfinite(d.qpos).all()))
    results[name]=driven
candidate=apply_edits(robot,fixture_edit(robot,'seed',0,0))
results['sensitivity']=[]
for friction in [.5,.65,.8]:
    r,t,xml=run(candidate,17,friction_override=friction)
    results['sensitivity'].append(r)
    (out/f'friction_{friction}.xml').write_text(xml);np.savez_compressed(out/f'friction_{friction}.npz',**t)
(out/'verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
