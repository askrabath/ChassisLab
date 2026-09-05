import os
import json
from pathlib import Path
import numpy as np


def add_commands(sub):
    for name in ['field','seed-robot']:
        p=sub.add_parser(name);p.add_argument('--output',default='assemblies/override_fixtures' if name=='field' else 'assemblies/claw_seed');p.add_argument('--viewer',action='store_true')
    p=sub.add_parser('mechanism-benchmark');p.add_argument('--design');p.add_argument('--seed',type=int,default=41);p.add_argument('--task',choices=['pin','cup'],default='pin');p.add_argument('--output',default='assemblies/single_benchmark')
    p=sub.add_parser('search');mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--ai',action='store_true');mode.add_argument('--offline',action='store_true')
    for name,default,choices in [('generations',3,range(1,11)),('candidates',3,range(1,11)),('trials',2,range(1,3)),('tuning',2,range(0,7)),('duration',32,range(10,121)),('api-calls',12,range(1,61)),('output-tokens',3000,range(256,8193)),('wall-seconds',600,range(30,3601))]:
        p.add_argument('--'+name,type=int,default=default,choices=choices,metavar='N')
    p.add_argument('--model',default=os.environ.get('CHASSIS_LAB_MODEL','gpt-5.6-sol'));p.add_argument('--reasoning',choices=['low','medium','high'],default='low');p.add_argument('--output',default='mechanism_runs');p.add_argument('--no-render',action='store_true')
    p=sub.add_parser('resume');p.add_argument('directory');p.add_argument('--no-render',action='store_true')
    p=sub.add_parser('mechanism-replay');p.add_argument('path');p.add_argument('--headless',action='store_true')
    p=sub.add_parser('compare');p.add_argument('directory')
    p=sub.add_parser('cad-view');p.add_argument('asset',choices=['clawbot','override_field']);p.add_argument('--viewer',action='store_true')
    p=sub.add_parser('cad-demo');p.add_argument('--output',default='assemblies/cad_drive');p.add_argument('--no-render',action='store_true');p.add_argument('--viewer',action='store_true')
    p.add_argument('--full-detail',action='store_true');p.add_argument('--full-field',action='store_true')


def dispatch(a):
    from .design import seed,Robot
    from .visual import replay
    from .search import search,save,report
    if a.command=='cad-view':
        from ..cad_scene import inspection,render
        output=inspection(a.asset)
        render(output,close=a.asset=='clawbot')
        if a.viewer:replay(output)
        return 0
    if a.command=='cad-demo':
        from ..cad_drive import run
        output=run(a.output,not a.no_render,not a.full_detail,not a.full_field)
        if a.viewer:replay(output)
        return 0
    if a.command in ['search','resume']:
        from dotenv import load_dotenv
        load_dotenv('.env.local',override=False);load_dotenv('.env',override=False)
        if a.command=='resume':root=search(resume=a.directory,render=not a.no_render)
        else:
            kwargs=vars(a).copy()
            for k in ['command','offline','no_render']:kwargs.pop(k)
            root=search(**kwargs,render=not a.no_render)
        return 0 if json.loads((root/'state.json').read_text())['status']=='complete' else 1
    if a.command=='mechanism-replay':replay(a.path,a.headless);return 0
    if a.command=='compare':
        root=Path(a.directory);report(root,json.loads((root/'state.json').read_text()));print((root/'report.html').resolve());return 0
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if a.command=='field':
        from .fixtures import validate
        validate(root)
        if a.viewer:replay(root/'full_field.xml')
    elif a.command=='seed-robot':
        import mujoco
        from .robot import compile_robot,initial_data,construction_check
        design=seed();xml=compile_robot(design);m=mujoco.MjModel.from_xml_string(xml);d=initial_data(m,design)
        (root/'seed.xml').write_text(xml);save(root/'design.json',design.model_dump());save(root/'construction.json',construction_check(m,d,design))
        np.savez_compressed(root/'seed.npz',qpos=np.array([d.qpos.copy()]))
        if a.viewer:replay(root/'seed.xml')
    else:
        from .sim import run
        design=Robot.model_validate_json(Path(a.design).read_text()) if a.design else seed()
        metrics,trajectory,xml=run(design,a.seed,a.task)
        save(root/'trial.json',metrics);save(root/'design.json',design.model_dump());(root/'trial.xml').write_text(xml);np.savez_compressed(root/'trial.npz',**trajectory);print(json.dumps(metrics,indent=2))
    print(root.resolve());return 0
