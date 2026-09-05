import mujoco
import numpy as np
from chassis_lab.assembly import build, settle, PITCH, RADIUS


def test_assembled_chassis_stands_on_four_wheels():
    xml,catalog=build()
    model=mujoco.MjModel.from_xml_string(xml)
    data,result=settle(model)
    assert result['standing'] and len(result['wheels_touching_floor'])==4
    assert result['root_speed_m_s']<1e-5
    assert model.nv==10 and model.nu==4
    assert len([p for p in catalog['parts'] if p['type']=='bearing flat'])==8
    assert len([p for p in catalog['parts'] if p['type']=='shaft collar'])==8
    # Beam underside clears the complete spinning tire envelope.
    assert 6*PITCH-1.5*PITCH > RADIUS+.005
    # Only the free root and four wheel axles introduce DOFs: screws are fixed.
    assert model.njnt==5
    assert np.isfinite(data.qpos).all()
