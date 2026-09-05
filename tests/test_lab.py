import math
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from pydantic import ValidationError
from chassis_lab.schema import Design, Position, sample_design
from chassis_lab.physics import compile_mjcf, initialize, motor_torques, yaw_of
from chassis_lab.benchmark import simulate, aggregate, compare


@pytest.mark.parametrize('change',[
    {'track_width_m':0.25}, {'wheelbase_m':0.32}, {'battery':{'x_m':0.18,'y_m':0}},
    {'ballast':{'x_m':0,'y_m':0}}, {'wheel_diameter_m':0.1}, {'gearing_rpm':300},
    {'frame_width_m':float('nan')}, {'mass_kg':1}, {'frame_length_m':0.5},
])
def test_invalid_designs(change):
    with pytest.raises(ValidationError):
        Design.model_validate(sample_design().model_dump()|change)


def test_generated_geometry_and_mass():
    d=sample_design()
    m=mujoco.MjModel.from_xml_string(compile_mjcf(d))
    assert m.nq==11 and m.nv==10 and m.nu==4
    assert sum(m.body_mass)==pytest.approx(d.total_mass_kg,abs=1e-12)
    assert m.geom_size[m.geom('frame').id,:2]==pytest.approx([d.frame_length_m/2,d.frame_width_m/2])
    for i in range(4):
        assert abs(m.body_pos[m.body(f'wheel_{i}').id,0])==pytest.approx(d.wheelbase_m/2)
        assert abs(m.body_pos[m.body(f'wheel_{i}').id,1])==pytest.approx(d.track_width_m/2)
        assert m.geom_size[m.geom(f'tire_{i}').id,0]==pytest.approx(d.wheel_diameter_m/2)
    assert np.all(m.body_inertia[1:]>0)
    shifted=Design.model_validate(d.model_dump()|{'battery':{'x_m':0.07,'y_m':0}})
    ms=mujoco.MjModel.from_xml_string(compile_mjcf(shifted))
    assert ms.body_ipos[1,0]>m.body_ipos[1,0]
    assert not np.allclose(ms.body_inertia[1],m.body_inertia[1])


def test_wheel_actuation_moves_and_turns():
    d=sample_design()
    for desired,turn in [(np.ones(4)*10,False),(np.array([-10,-10,10,10]),True)]:
        m,s,pose,_=initialize(d,11)
        for _ in range(1000):
            s.ctrl[:]=motor_torques(d,s.qvel[6:],desired)[0]
            mujoco.mj_step(m,s)
        assert np.isfinite(s.qpos).all() and np.isfinite(s.qvel).all()
        assert s.qpos[2]==pytest.approx(d.wheel_diameter_m/2,abs=0.005)
        if turn: assert abs(yaw_of(s.qpos[3:7])-pose[2])>0.5
        else: assert s.qpos[0]-pose[0]>0.7
    m,s,pose,_=initialize(d,11)
    for _ in range(1000): mujoco.mj_step(m,s)
    assert np.linalg.norm(s.qpos[:2]-pose[:2])<0.002


def test_motor_bounds_and_back_emf():
    d=sample_design(); free=d.gearing_rpm*2*math.pi/60
    t,_=motor_torques(d,np.zeros(4),np.ones(4)*1000)
    assert np.max(np.abs(t))<=d.stall_torque_nm
    t,_=motor_torques(d,np.ones(4)*free,np.ones(4)*free)
    assert t==pytest.approx(np.zeros(4))
    t,_=motor_torques(d,np.ones(4)*free,np.zeros(4))
    assert np.all(t<0)


def test_full_course_and_ranking_rejects_shortcuts():
    success,_,_=simulate(sample_design(),11)
    stopped,_,_=simulate(sample_design(),11,duration=1,stationary=True)
    assert success.completed and success.collision_episodes==0 and not success.unstable
    assert success.braking_stop_time_s>=0.35
    assert success.waypoints_reached==6 and success.waypoint_progress==1
    assert success.parking_heading_error_rad<0.1
    assert not stopped.completed and stopped.waypoint_progress<1
    assert compare(aggregate([stopped]),aggregate([success]))=='improvement'
    shortcut=stopped.model_copy(update={'waypoint_progress':0.999,'rms_tracking_error_m':0.,'elapsed_s':0.})
    assert compare(aggregate([shortcut]),aggregate([success]))=='improvement'
    colliding=success.model_copy(update={'collision_episodes':1,'completion_time_s':0.1})
    assert compare(aggregate([success]),aggregate([colliding]))=='regression'


def test_seeded_reproducibility():
    a,ta,_=simulate(sample_design(),23,duration=3)
    b,tb,_=simulate(sample_design(),23,duration=3)
    assert a.model_dump()==b.model_dump()
    np.testing.assert_allclose(ta['qpos'],tb['qpos'],rtol=0,atol=1e-10)
    c,_,_=simulate(sample_design(),37,duration=0.1)
    assert a.initial_pose!=c.initial_pose and a.friction!=c.friction


def test_ai_repair_is_bounded_and_usage_preserved():
    from chassis_lab.ai import Designer, WireProposal, ProposalFailure
    bad=sample_design().model_dump()|{'track_width_m':0.25}
    calls=[]; saved={}
    def parse(**kwargs):
        calls.append(kwargs)
        output=WireProposal.model_validate(dict(design=bad,rationale='x',predicted_benefit='x',hypothesis='x'))
        return SimpleNamespace(id='test',status='completed',usage=SimpleNamespace(model_dump=lambda:{'total_tokens':10}),
                               output_text=output.model_dump_json(),output_parsed=output)
    designer=Designer('fixture-test',lambda k,v:saved.update({k:v}),{},client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    with pytest.raises(ProposalFailure): designer.propose()
    assert len(calls)==2
    assert all(c['max_output_tokens']==2500 for c in calls)
    assert saved['ai_calls.json'][0]['usage']['total_tokens']==10
    assert 'validation_error' in saved['ai_calls.json'][1]
