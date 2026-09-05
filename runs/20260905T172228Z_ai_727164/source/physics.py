"""Primitive rigid bodies, four hinge actuators, and ordinary frictional contact."""
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .schema import Design

DT = 0.002
WAYPOINTS = np.array([[0,0], [2,0], [2.65,0.60], [3.45,-0.60], [4.25,0.60], [5.1,0], [5.65,0]])
OBSTACLES = [(2.65,-0.30), (3.45,0.30), (4.25,-0.30)]
DEV_SEEDS = (11, 23, 37)
FINAL_SEEDS = (101, 211, 307)


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k:str(v) for k,v in attrs.items()})


def compile_mjcf(design: Design, friction=0.7):
    d = Design.model_validate(design.model_dump())
    root = ET.Element("mujoco", model="chassis_lab")
    element(root, "compiler", angle="radian", inertiafromgeom="true")
    element(root, "option", timestep=DT, gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic", iterations=60)
    default = element(root,"default")
    element(default,"geom", friction=f"{friction} 0.002 0.0001", condim=3, solref="0.01 1", solimp="0.95 0.99 0.001")
    visual=element(root,"visual")
    element(visual,"global", offwidth=1000, offheight=700)
    world = element(root,"worldbody")
    element(world,"light", pos="2 0 5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    element(world,"geom", name="floor", type="plane", size="8 5 0.1", rgba="0.15 0.19 0.23 1")
    for i,(x,y,sx,sy) in enumerate([(-0.8,0,0.05,1.8),(6.5,0,0.05,1.8),(2.85,-1.8,3.7,0.05),(2.85,1.8,3.7,0.05)]):
        element(world,"geom", name=f"wall_{i}", type="box", pos=f"{x} {y} 0.15", size=f"{sx} {sy} 0.15", rgba="0.4 0.45 0.5 1")
    for i,(x,y) in enumerate(OBSTACLES):
        element(world,"geom", name=f"obstacle_{i}", type="cylinder", pos=f"{x} {y} 0.16", size="0.14 0.16", rgba="0.96 0.4 0.14 1")
    for i,(x,y) in enumerate(WAYPOINTS[1:]):
        element(world,"site", name=f"waypoint_{i}", type="cylinder", pos=f"{x} {y} 0.004", size="0.08 0.003", rgba="0.25 0.8 0.8 0.5")
    element(world,"site", name="parking", type="box", pos="5.65 0 0.003", size="0.3 0.3 0.002", rgba="0.2 0.9 0.5 0.4")
    r=d.wheel_diameter_m/2
    body=element(world,"body", name="chassis", pos=f"0 0 {r+0.004}")
    element(body,"freejoint", name="root")
    element(body,"geom", name="frame", type="box", size=f"{d.frame_length_m/2} {d.frame_width_m/2} 0.0125", mass=d.frame_mass_kg, rgba="0.25 0.5 0.85 1")
    element(body,"geom", name="battery", type="box", pos=f"{d.battery.x_m} {d.battery.y_m} 0.035", size="0.06 0.0325 0.0225", mass=0.35, rgba="0.85 0.23 0.25 1")
    if d.ballast:
        element(body,"geom", name="ballast", type="box", pos=f"{d.ballast.x_m} {d.ballast.y_m} 0.0275", size="0.025 0.025 0.015", mass=0.30, rgba="0.7 0.7 0.7 1")
    for i,(x,y) in enumerate([(d.wheelbase_m/2,d.track_width_m/2),(-d.wheelbase_m/2,d.track_width_m/2),(d.wheelbase_m/2,-d.track_width_m/2),(-d.wheelbase_m/2,-d.track_width_m/2)]):
        # Fixed motor mass on underside; component volumes do not intersect battery.
        element(body,"geom", name=f"motor_mass_{i}", type="box", pos=f"{x} {math.copysign(d.frame_width_m/2-0.02,y)} -0.02", size="0.018 0.015 0.0075", mass=0.28, rgba="0.3 0.3 0.3 1")
        wheel=element(body,"body", name=f"wheel_{i}", pos=f"{x} {y} 0")
        element(wheel,"joint", name=f"axle_{i}", type="hinge", axis="0 1 0", damping=0.001, armature=0.0001)
        element(wheel,"geom", name=f"tire_{i}", type="cylinder", size=f"{r} 0.0125", quat="0.7071067812 0.7071067812 0 0", mass=0.12, rgba="0.055 0.06 0.065 1")
        element(wheel,"site", type="sphere", pos=f"{r*0.7} 0.013 0", size="0.006", rgba="1 1 1 1")
    actuator=element(root,"actuator")
    for i in range(4):
        element(actuator,"motor", name=f"drive_{i}", joint=f"axle_{i}", gear=1, ctrllimited="true", ctrlrange=f"{-d.stall_torque_nm} {d.stall_torque_nm}")
    return ET.tostring(root,encoding="unicode")


def motor_torques(design, velocity, desired_velocity):
    """Voltage proportional speed feedback + back EMF; symmetric current limit."""
    free = design.gearing_rpm*2*math.pi/60
    voltage=np.clip(desired_velocity/free + 0.8*(desired_velocity-velocity),-1,1)
    raw=design.stall_torque_nm*(voltage-velocity/free)
    return np.clip(raw,-design.stall_torque_nm,design.stall_torque_nm), np.abs(voltage)>=0.999


def yaw_of(quat):
    w,x,y,z=quat
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def wrap(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


def initialize(design, seed, xml=None):
    rng=np.random.default_rng(seed)
    pose=[float(rng.uniform(-0.025,0.025)),float(rng.uniform(-0.025,0.025)),float(rng.uniform(-0.04,0.04))]
    friction=float(rng.uniform(0.55,0.80))
    model=mujoco.MjModel.from_xml_string(xml or compile_mjcf(design,friction))
    data=mujoco.MjData(model)
    data.qpos[:2]=pose[:2]
    data.qpos[3:7]=[math.cos(pose[2]/2),0,0,math.sin(pose[2]/2)]
    mujoco.mj_forward(model,data)
    return model,data,pose,friction
