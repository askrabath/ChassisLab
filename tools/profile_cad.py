"""Same-machine rendering/physics comparison; timings are observational."""
import json
import time
from pathlib import Path
import mujoco
import numpy as np


def profile(path):
    t=time.perf_counter();m=mujoco.MjModel.from_xml_path(str(path.resolve()));load=time.perf_counter()-t
    d=mujoco.MjData(m);t=time.perf_counter()
    for _ in range(5000):mujoco.mj_step(m,d)
    physics=time.perf_counter()-t
    cam=mujoco.MjvCamera();cam.lookat[:]=[-.475,-.9,.1];cam.distance=2.3;cam.elevation=-40
    timings=[]
    with mujoco.Renderer(m,height=600,width=800) as renderer:
        for i in range(13):
            t=time.perf_counter();renderer.update_scene(d,camera=cam);renderer.render()
            if i>=3:timings.append(time.perf_counter()-t)
    triangles=sum(int(m.mesh_facenum[m.geom_dataid[i]]) for i in range(m.ngeom)
                  if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[i]<3)
    return dict(load_s=load,physics_10s_wall_s=physics,render_median_ms=1000*float(np.median(timings)),
                render_fps=1/float(np.median(timings)),visible_mesh_triangles=triangles,geoms=m.ngeom,
                renderer='800x600 offscreen; 3 warmup + 10 timed frames; update_scene and readback included')


def main():
    full=Path('assemblies/cad_performance/full/cad_drive.xml');fast=Path('assemblies/cad_drive/cad_drive.xml')
    results={'full':profile(full),'optimized':profile(fast)}
    a=np.load(full.with_suffix('.npz'))['qpos'];b=np.load(fast.with_suffix('.npz'))['qpos']
    results['trajectory_max_abs_difference']=float(np.max(abs(a-b)))
    results['render_speedup']=results['full']['render_median_ms']/results['optimized']['render_median_ms']
    Path('assemblies/cad_performance/comparison.json').write_text(json.dumps(results,indent=2))
    import html
    Path('assemblies/cad_performance/report.html').write_text('<!doctype html><meta charset="utf-8"><title>CAD rendering comparison</title><style>body{font:16px system-ui;max-width:960px;margin:40px auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>CAD rendering comparison</h1><p>Same machine, camera and 800 × 600 resolution. Timing includes drawing and pixel readback; interactive viewer performance may differ. Physics parameters are unchanged for retained objects.</p><pre>'+html.escape(json.dumps(results,indent=2))+'</pre><img src="../cad_drive/field.png"><p>Arm/claw and field objects remain fixed in this driving test. This optimization does not add scoring or improve the existing contact calibration.</p>')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
