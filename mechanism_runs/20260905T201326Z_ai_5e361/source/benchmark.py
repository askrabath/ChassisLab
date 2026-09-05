import math
import numpy as np
import mujoco
from .physics import DT, WAYPOINTS, initialize, motor_torques, yaw_of, wrap
from .schema import Trial

TIME_LIMIT = 45.0


def segment_distance(p, a, b):
    frac=float(np.clip(np.dot(p-a,b-a)/np.dot(b-a,b-a),0,1))
    return float(np.linalg.norm(p-(a+frac*(b-a)))), frac


class Controller:
    """Identical SI-unit feedback for every candidate; no candidate-specific tuning."""
    def __init__(self, design):
        self.design=design
        self.target=1
        self.stop_started=None
        self.braking_stop_time=None
        self.hold=0.0
        self.done=False

    def command(self, data):
        p=data.qpos[:2]
        yaw=yaw_of(data.qpos[3:7])
        speed=float(np.linalg.norm(data.qvel[:2]))
        delta=WAYPOINTS[self.target]-p
        distance=float(np.linalg.norm(delta))
        bearing=wrap(math.atan2(delta[1],delta[0])-yaw)
        v=min(0.85,1.8*distance)*max(0,1-abs(bearing)/0.40)
        omega=float(np.clip(3.2*bearing,-1.8,1.8))
        if self.target==1 and distance<0.09:
            if self.stop_started is None:
                self.stop_started=float(data.time)
            v=omega=0
            self.hold = self.hold+DT if speed<0.045 else 0
            if self.hold>=0.35:
                self.braking_stop_time=float(data.time-self.stop_started)
                self.target+=1
                self.hold=0
        elif self.target==len(WAYPOINTS)-1 and distance<0.085:
            v=0
            omega=float(np.clip(-3.2*yaw,-1.2,1.2))
            self.hold=self.hold+DT if abs(yaw)<0.10 and speed<0.045 and abs(data.qvel[5])<0.10 else 0
            self.done=self.hold>=0.5
        elif self.target not in (1,len(WAYPOINTS)-1) and distance<0.11:
            self.target+=1
        r=self.design.wheel_diameter_m/2
        t=self.design.track_width_m
        desired=np.array([v-omega*t/2]*2+[v+omega*t/2]*2)/r
        free=self.design.gearing_rpm*2*math.pi/60
        desired*=min(1,free/max(float(np.max(np.abs(desired))),1e-6))
        return desired


def simulate(design, seed, *, duration=TIME_LIMIT, stationary=False):
    model,data,pose,friction=initialize(design,seed)
    controller=Controller(design)
    errors=[]; trace=[]; qpos=[]; times=[]
    travel=np.zeros(4); prev=data.qpos.copy(); distance=0.; peak=0.; saturation=0
    episodes=0; last_contact={}; failures=[]; unstable=False
    hazard_ids={i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(('obstacle_','wall_'))}
    for step in range(round(duration/DT)):
        desired=controller.command(data)
        torque,sat=motor_torques(design,data.qvel[6:],desired)
        data.ctrl[:]=0 if stationary else torque
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.any(data.warning.number):
            unstable=True; failures.append('nonfinite state or MuJoCo numerical warning'); break
        tilt=1-2*(data.qpos[4]**2+data.qpos[5]**2)
        if tilt<math.cos(math.radians(35)) or data.qpos[2]>0.30 or np.max(np.abs(data.qvel))>500:
            unstable=True; failures.append('tilt >35 degrees, airborne >0.30 m, or excessive velocity'); break
        contact_now=set()
        for contact in data.contact:
            if contact.dist>0: continue
            for a,b in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
                if a in hazard_ids and model.geom_bodyid[b]!=0:
                    contact_now.add(a)
        for hazard in contact_now:
            if data.time-last_contact.get(hazard,-1e9)>0.25:
                episodes+=1
            last_contact[hazard]=float(data.time)
        travel+=np.abs(data.qpos[7:]-prev[7:])*design.wheel_diameter_m/2
        distance+=float(np.linalg.norm(data.qpos[:2]-prev[:2]))
        peak=max(peak,float(np.linalg.norm(data.qvel[:2])))
        prev=data.qpos.copy()
        saturation+=int(np.count_nonzero(sat)) if not stationary else 0
        error,_=segment_distance(data.qpos[:2],WAYPOINTS[controller.target-1],WAYPOINTS[controller.target])
        errors.append(error)
        if step%10==0 or controller.done:
            trace.append([float(data.time),*data.qpos[:2].tolist(),yaw_of(data.qpos[3:7]),controller.target])
            qpos.append(data.qpos.copy()); times.append(float(data.time))
        if controller.done: break
    completed=controller.done and not unstable
    if not completed: failures.append('course incomplete')
    if episodes: failures.append(f'{episodes} obstacle/wall collision episodes')
    _,fraction=segment_distance(data.qpos[:2],WAYPOINTS[controller.target-1],WAYPOINTS[controller.target])
    progress=1.0 if completed else min(0.999,(controller.target-1+fraction)/(len(WAYPOINTS)-1))
    result=Trial(seed=seed, friction=friction, initial_pose=pose, completed=completed,
        completion_time_s=float(data.time) if completed else None,elapsed_s=float(data.time),
        waypoint_progress=progress,waypoints_reached=len(WAYPOINTS)-1 if completed else controller.target-1,
        collision_episodes=episodes,rms_tracking_error_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else 0,
        parking_position_error_m=float(np.linalg.norm(data.qpos[:2]-WAYPOINTS[-1])),
        parking_heading_error_rad=abs(wrap(yaw_of(data.qpos[3:7]))),unstable=unstable,failures=failures,
        wheel_travel_m=travel.tolist(),saturation_fraction=saturation/((step+1)*4),distance_traveled_m=distance,
        peak_speed_m_s=peak,braking_stop_time_s=controller.braking_stop_time)
    return result,dict(trace=np.array(trace),qpos=np.array(qpos),time=np.array(times)),model


def aggregate(trials):
    n=len(trials)
    clean=sum(t.completed and not t.collision_episodes and not t.unstable for t in trials)
    complete=sum(t.completed for t in trials)
    # Lexicographic, lower is better. Incomplete runs cannot buy rank with low time.
    key=[-clean,-complete,sum(t.unstable for t in trials),sum(t.collision_episodes for t in trials),
         -float(np.mean([t.waypoint_progress for t in trials])),
         float(np.mean([t.completion_time_s if t.completed else TIME_LIMIT for t in trials])),
         float(np.mean([t.rms_tracking_error_m for t in trials]))]
    return dict(trials=n,clean_completions=clean,completions=complete,ranking_key=key,
                mean_time_with_timeout_s=key[5],mean_tracking_error_m=key[6],
                mean_progress=-key[4],collision_episodes=key[3],unstable_runs=key[2],
                failures=[dict(seed=t.seed,failures=t.failures) for t in trials if t.failures])


def compare(baseline,candidate):
    a=baseline['ranking_key']; b=candidate['ranking_key']
    # Speed differences <=0.10 s and tracking <=1 mm are treated as ties.
    for i,(x,y) in enumerate(zip(a,b)):
        tolerance=0.10 if i==5 else (0.001 if i in (4,6) else 0)
        if abs(x-y)>tolerance:
            return 'improvement' if y<x else 'regression'
    return 'inconclusive'
