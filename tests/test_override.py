import numpy as np
import mujoco
import pytest
from pydantic import ValidationError
from chassis_lab.override.design import seed, Robot, fixture_edit, apply_edits
from chassis_lab.override.robot import compile_robot, initial_data, construction_check
from chassis_lab.override.sim import run, rank, Motors
from chassis_lab.override.fixtures import validate


def roller():
    r=seed();return apply_edits(r,fixture_edit(r,'seed',0,0))


@pytest.mark.parametrize('mutation', ['disconnected','axis','catalog','duplicate','nan','fixed_motor','gear'])
def test_assembly_rejects_invalid(mutation):
    d=seed().model_dump();p=d['parts'][0]
    if mutation=='disconnected':p['parent']='missing'
    if mutation=='axis':p['joint']['axis']=[0,0,0]
    if mutation=='catalog':p['catalog']='magic_gripper'
    if mutation=='duplicate':d['parts'][1]['id']=p['id']
    if mutation=='nan':p['offset_m'][0]=float('nan')
    if mutation=='fixed_motor':p['joint']['kind']='fixed'
    if mutation=='gear':p['joint']['reduction']=999
    with pytest.raises(ValidationError):Robot.model_validate(d)


def test_structural_edit_mass_and_envelope():
    r=roller();assert r.architecture!=seed().architecture
    m=mujoco.MjModel.from_xml_string(compile_robot(r));d=initial_data(m,r)
    c=construction_check(m,d,r)
    assert max(c['initial_envelope_m'])<.4582
    assert 3<c['mass_kg']<6
    assert c['mass_kg']<sum(m.body_mass)
    assert not c['official_legality']
    assert m.nu==8


def test_contact_fixtures(tmp_path):
    r=validate(tmp_path)
    assert all(x['finite'] and x['warnings']==0 for x in r.values() if 'finite' in x)
    assert r['pin_goal']['pin_placed']
    assert r['pin_cup_stack']['stable_stack']
    assert not r['outside_goal']['pin_placed']
    assert all(r['scoring_negative_tests'].values())


def test_actuated_seed_moves_arm_claw_and_drives():
    r=seed();m=mujoco.MjModel.from_xml_string(compile_robot(r));d=initial_data(m,r);c=Motors(m,r)
    start=d.qpos.copy()
    for _ in range(1000):c.apply(d,.2,0,.1,.5);mujoco.mj_step(m,d)
    assert d.qpos[0]>start[0]+.15
    for name in ['arm_joint','left_finger_joint','right_finger_joint']:
        i=m.joint(name).qposadr[0];assert abs(d.qpos[i]-start[i])>.15
    assert np.isfinite(d.qpos).all()
    assert np.all(np.abs(d.ctrl)<=m.actuator_ctrlrange[:,1]+1e-9)


def test_physical_manipulation_reproducibility_and_ranking():
    r=roller();a,t,_=run(r,17);b,u,_=run(r,17)
    assert a['success'] and a['picked_up'] and not a['unstable']
    assert a['peak_object_height_m']>.2
    np.testing.assert_allclose(t['qpos'],u['qpos'],atol=1e-9,rtol=0)
    assert a==b
    stationary=dict(a,success=False,time_s=0,picked_up=False)
    assert rank([a])<rank([stationary])


def test_floor_friction_is_not_masked_by_geom_mixing():
    r=seed();m=mujoco.MjModel.from_xml_string(compile_robot(r,friction=.5));d=initial_data(m,r)
    for _ in range(200):mujoco.mj_step(m,d)
    floor=m.geom('floor').id
    contacts=[c for c in d.contact if floor in (c.geom1,c.geom2)]
    assert contacts
    assert all(abs(c.friction[0]-.5)<1e-10 for c in contacts)


def test_unvalidated_powered_slide_rejected():
    d=seed().model_dump();d['parts'][0]['joint'].update(kind='prismatic',limits=[-.05,.05])
    with pytest.raises(ValidationError,match='linear controller'):Robot.model_validate(d)


def test_fixed_passive_part_accepts_zero_limits():
    robot=seed()
    proposal=fixture_edit(robot,'seed',0,0)
    revised=apply_edits(robot,proposal)
    guide=next(p for p in revised.parts if p.catalog=='guide')
    guide.joint.limits=[0,0]
    Robot.model_validate(revised.model_dump())
