from pathlib import Path
import mujoco
import numpy as np
import pytest
from chassis_lab.cad_drive import compile_scene, simulate
from chassis_lab.cad_scene import ROOT, tree

pytestmark=pytest.mark.skipif(not (ROOT/'data/cad_assets/override_field/assembly_tree.json').exists(),reason='Optional supplied CAD assets are not imported')


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    return compile_scene(tmp_path_factory.mktemp('cad'))


def test_import_instances_and_physical_wheel_geometry(scene):
    path,config=scene
    assert len(tree('clawbot')['instances'])==143
    assert len(tree('override_field')['instances'])==1120
    assert len(config['wheels'])==4
    assert sum(w['driven'] for w in config['wheels'])==2
    centers=np.array([w['center_m'] for w in config['wheels']])
    assert np.ptp(centers[:,0])==pytest.approx(.1778503,abs=1e-5)
    m=mujoco.MjModel.from_xml_path(str(path))
    assert m.nu==2
    assert m.body_mass.sum()==pytest.approx(2.83)
    assert m.jnt_type[0]==mujoco.mjtJoint.mjJNT_FREE


def test_actuated_motion_turning_stability_and_repeatability(scene):
    path,config=scene
    first=simulate(path,config);second=simulate(path,config)
    assert first['straight_displacement_m']>.5
    assert abs(first['turn_heading_change_deg'])>30
    assert first['max_tilt_deg']<10
    assert first['finite']
    assert np.allclose(first['final_position_m'],second['final_position_m'],atol=1e-8,rtol=0)
    # Control experiment: without wheel torques, gravity alone does not drive.
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    initial=d.qpos[:2].copy()
    for _ in range(1500):mujoco.mj_step(m,d)
    assert np.linalg.norm(d.qpos[:2]-initial)<.03


def test_visual_optimization_preserves_dynamics_and_contact_meshes(scene,tmp_path):
    fast_path,fast_config=scene
    full_path,full_config=compile_scene(tmp_path/'full',lightweight=False,crop=False)
    fast=mujoco.MjModel.from_xml_path(str(fast_path));full=mujoco.MjModel.from_xml_path(str(full_path))
    for name in ['body_mass','body_inertia','jnt_axis','actuator_ctrlrange']:
        assert np.allclose(getattr(fast,name),getattr(full,name),atol=1e-12,rtol=0)
    assert fast.opt.timestep==full.opt.timestep
    assert fast.ngeom<full.ngeom/2
    for i in range(fast.ngeom):
        if fast.geom_contype[i]==0:continue
        name=mujoco.mj_id2name(fast,mujoco.mjtObj.mjOBJ_GEOM,i)
        j=full.geom(name).id
        assert np.allclose(fast.geom_friction[i],full.geom_friction[j])
        if fast.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            a,b=fast.geom_dataid[i],full.geom_dataid[j]
            assert fast.mesh_facenum[a]==full.mesh_facenum[b]
    simulate(full_path,full_config);simulate(fast_path,fast_config)
    assert np.allclose(np.load(full_path.with_suffix('.npz'))['qpos'],np.load(fast_path.with_suffix('.npz'))['qpos'],atol=1e-8,rtol=0)
