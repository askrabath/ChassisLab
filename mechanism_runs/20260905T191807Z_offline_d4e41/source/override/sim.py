"""Privileged-state finite-state controller; only bounded motor torques act."""
import math
import mujoco
import numpy as np
from .robot import compile_robot,initial_data,construction_check
from . import field
from ..physics import yaw_of,wrap

DT=.002
TRAIN=(17,29); DEVELOPMENT=(41,53); FINAL=(1009,1013,1019)


class Motors:
    def __init__(self,m,design):
        self.m=m;self.design=design;self.integral=np.zeros(m.nu)
        self.targets={p.joint.role:(.9 if p.joint.role=='lift' else -.9 if p.joint.role=='wrist' else 0) for p in design.parts if p.joint.role!='passive'}
        self.joints={p.joint.role:p for p in design.parts if p.joint.role!='passive'}
        self.saturated=0;self.steps=0
    def apply(self,d,v,omega,arm_angle,closure,acquire=True):
        yaw=yaw_of(d.qpos[3:7])
        desired=np.array([v-omega*.2413/2]*2+[v+omega*.2413/2]*2)/.0508
        for i in range(4):
            vel=d.qvel[self.m.jnt_dofadr[self.m.actuator_trnid[i,0]]]
            err=desired[i]-vel
            self.integral[i]=np.clip(self.integral[i]+err*DT,-.8,.8)
            u=np.clip(desired[i]/20.944+.12*err+.7*self.integral[i],-1,1)
            d.ctrl[i]=np.clip(1.05*(u-vel/20.944),-1.05,1.05)
            self.saturated+=abs(u)>.999
        for role,p in self.joints.items():
            aid=self.m.actuator(role).id;jid=self.m.joint(p.id+'_joint').id
            q=d.qpos[self.m.jnt_qposadr[jid]];vel=d.qvel[self.m.jnt_dofadr[jid]]
            bound=float(self.m.actuator_ctrlrange[aid,1]);free=20.944/p.joint.reduction
            if role.startswith('roller'):
                desired_vel=(-12 if role=='roller_left' else 12) if acquire else 0
                u=np.clip(desired_vel/free+.2*(desired_vel-vel),-1,1)
            else:
                goal=arm_angle if role=='lift' else -arm_angle if role=='wrist' else (-closure if role=='grip_left' else closure)
                self.targets[role]+=float(np.clip(goal-self.targets[role],-.6*DT,.6*DT))
                error=self.targets[role]-q
                self.integral[aid]=np.clip(self.integral[aid]+error*DT,-.25,.25)
                u=np.clip(6*error+4*self.integral[aid]-.35*vel,-1,1)
            d.ctrl[aid]=np.clip(bound*(u-vel/free),-bound,bound)
            self.saturated+=abs(u)>.999
        self.steps+=1


def run(design,seed,task='pin',duration=32,closure=.62,friction_override=None):
    rng=np.random.default_rng(seed)
    xy=(.36+float(rng.uniform(-.008,.008)),float(rng.uniform(-.008,.008)))
    friction=float(rng.uniform(.5,.8)) if friction_override is None else friction_override
    xml=compile_robot(design,task,friction,xy)
    if task=='cup':
        import xml.etree.ElementTree as ET
        root=ET.fromstring(xml);field.pin(root.find('asset'),root.find('worldbody'),'support_pin',(.80,0,.01+field.PIN_H/2+.002))
        xml=ET.tostring(root,encoding='unicode')
    m=mujoco.MjModel.from_xml_string(xml);d=initial_data(m,design);checks=construction_check(m,d,design)
    control=Motors(m,design);phase='approach';phase_start=0.;history=[]
    arm=next(p for p in design.parts if p.joint.role=='lift')
    obj=m.body('object').id;grasp=m.site('grasp_center').id
    rootid=m.body('chassis').id
    picked=False;lifted=False;dropped=False;collisions=0;lastcollision=-99.;unstable=False
    acquired_at=None;placement=None;release_at=None;hold=0.;peak_z=0.;wheeltravel=np.zeros(4);priorwheel=d.qpos[7:11].copy()
    trajectory=[];metrics=[];times=[];best_error=99.;recovery=0
    goalxy=np.array([.8,0]);start_obj_z=field.PIN_H/2 if task=='pin' else field.CUP_H/2
    gripheight=.13 if task=='pin' else .115
    def transition(nextphase):
        nonlocal phase,phase_start
        history.append(dict(time_s=float(d.time),from_phase=phase,to_phase=nextphase))
        phase=nextphase;phase_start=float(d.time)
    for step in range(int(duration/DT)):
        pos=d.xpos[obj].copy();gp=d.site_xpos[grasp].copy();yaw=yaw_of(d.qpos[3:7])
        contact=False;hazard=False
        for c in d.contact:
            if c.dist>0:continue
            a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            roots=(m.body_rootid[a],m.body_rootid[b])
            if obj in roots and rootid in roots:contact=True
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in (c.geom1,c.geom2)]
            if rootid in roots and any(n.startswith('target_goal') for n in names):hazard=True
        if hazard:
            if d.time-lastcollision>.25:collisions+=1
            lastcollision=d.time
        v=omega=0.;height=gripheight;close=-.25 if phase=='approach' else closure
        if task=='cup' and close>0:close=max(.25,closure-.18)
        if phase=='approach':
            delta=np.array(xy)-gp[:2];dist=float(np.linalg.norm(delta));best_error=min(best_error,dist)
            desired_yaw=math.atan2(xy[1]-d.qpos[1],xy[0]-d.qpos[0])
            omega=float(np.clip(3*wrap(desired_yaw-yaw),-1,1))
            v=float(np.clip(np.dot(delta,[math.cos(yaw),math.sin(yaw)])*2,-.15,.22)) if abs(wrap(desired_yaw-yaw))<.3 else 0
            if dist<.006 and abs(gp[2]-height)<.012 and d.time>2.5:transition('close')
        elif phase=='close':
            if d.time-phase_start>2.5:transition('lift')
        elif phase=='lift':
            height=.30
            if pos[2]>start_obj_z+.04 and contact:
                picked=True;lifted=True
                if acquired_at is None:acquired_at=float(d.time)
            if d.time-phase_start>4:
                if lifted:transition('transport')
                elif recovery<1:
                    recovery+=1;transition('approach')
                else:transition('failed_pickup')
        elif phase=='transport':
            height=.30;delta=goalxy-gp[:2]
            omega=float(np.clip(3*wrap(math.atan2(goalxy[1]-d.qpos[1],goalxy[0]-d.qpos[0])-yaw),-1,1))
            v=float(np.clip(np.dot(delta,[math.cos(yaw),math.sin(yaw)])*1.5,-.1,.22))
            if np.linalg.norm(delta)<.006 and np.linalg.norm(d.qvel[:2])<.03:transition('lower')
            if pos[2]<start_obj_z+.015:dropped=True
        elif phase=='lower':
            height=.153 if task=='pin' else .231
            if abs(gp[2]-height)<.015 and d.time-phase_start>3:transition('release');release_at=float(d.time)
        elif phase in ('release','settle'):
            close=-.25;height=.153 if task=='pin' else .231
            if phase=='release' and d.time-phase_start>1.5:transition('settle')
            if phase=='settle':
                height=.32;v=-.1 if d.time-phase_start<1.5 else 0
        elif phase=='failed_pickup':break
        angle=math.asin(float(np.clip((height-d.qpos[2]-arm.offset_m[2])/arm.size_m[0],-.99,.99)))
        angle=float(np.clip(angle,*arm.joint.limits))
        control.apply(d,v,omega,angle,close,acquire=phase not in ('release','settle','approach'))
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number) or 1-2*(d.qpos[4]**2+d.qpos[5]**2)<math.cos(.6):
            unstable=True;break
        wheeltravel+=np.abs(d.qpos[7:11]-priorwheel)*.0508;priorwheel=d.qpos[7:11].copy()
        peak_z=max(peak_z,float(d.xpos[obj,2]))
        if task=='pin':
            nested=field.nesting(d.xpos[obj],d.xmat[obj].reshape(3,3)[:,2],goalxy,field.GOAL_H,.010)
        else:
            support=m.body('support_pin').id
            # Cup lower opening encloses the upper half of an already goal-nested Pin.
            nested=bool(field.nesting(d.xpos[support],d.xmat[support].reshape(3,3)[:,2],goalxy,field.GOAL_H,.010)
                        and abs(d.xmat[obj].reshape(3,3)[2,2])>.90
                        and np.linalg.norm(d.xpos[obj,:2]-goalxy)<.012
                        and d.xpos[obj,2]-field.CUP_H/2 < d.xpos[support,2]+field.PIN_H/2 < d.xpos[obj,2])
        if nested and release_at is not None and not contact and phase=='settle':hold+=DT
        else:hold=0
        if step%25==0:
            trajectory.append(d.qpos.copy());times.append(float(d.time));metrics.append([*d.xpos[obj],*d.qpos[:3],*d.site_xpos[grasp]])
        if hold>=.75:placement=float(d.time);break
    failures=[]
    if not picked:failures.append('pickup_failed')
    elif placement is None:failures.append('placement_failed')
    if dropped:failures.append('dropped_during_transport')
    if collisions:failures.append('goal_collision')
    if unstable:failures.append('unstable')
    success=bool(placement is not None and picked and not unstable and not collisions)
    result=dict(timeout_s=duration,seed=seed,task=task,success=success,time_s=placement,elapsed_s=float(d.time),picked_up=picked,
        placed_subset=bool(placement is not None),phase=phase,failures=failures,drop=dropped,jam=phase=='failed_pickup',
        recovery_attempts=recovery,pickup_time_s=acquired_at,pickup_best_error_m=best_error,
        placement_error_m=float(np.linalg.norm(d.xpos[obj,:2]-goalxy)),peak_object_height_m=peak_z,
        collision_episodes=collisions,unstable=unstable,saturation_fraction=float(control.saturated/max(1,control.steps*m.nu)),
        wheel_travel_m=wheeltravel.tolist(),friction=friction,initial_object_xy=xy,controller={'closure':closure,'privileged_state':True},
        rule_violations=['collision is a benchmark failure, not automatically a game violation'] if collisions else [],
        official_match_score=None,construction=checks,transitions=history)
    return result,dict(qpos=np.array(trajectory),time=np.array(times),state=np.array(metrics)),xml


def rank(trials):
    return [-sum(t['success'] for t in trials),sum(t['unstable'] for t in trials),sum(t['collision_episodes'] for t in trials),
            sum(t['time_s'] if t['success'] else t.get('timeout_s',32) for t in trials)/len(trials),-sum(t['picked_up'] for t in trials)]
