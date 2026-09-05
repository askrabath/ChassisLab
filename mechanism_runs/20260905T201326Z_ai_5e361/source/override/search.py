"""Checkpointed population experiment. Agent output is data, never executable code."""
import base64
import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import mujoco
import numpy as np
from .design import Robot,Proposal,seed,apply_edits,fixture_edit
from .robot import compile_robot,initial_data,construction_check
from .sim import run,rank,TRAIN,DEVELOPMENT,FINAL


def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def hashcode():
    return {str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parents[1].rglob('*.py')}


def proposal_ai(parents,model,reasoning,root,limits):
    from openai import OpenAI
    client=OpenAI(max_retries=0,timeout=min(45,limits['wall_seconds']))
    path=root/'api_calls.json';calls=json.loads(path.read_text()) if path.exists() else []
    context=[];images=[]
    for parent in parents:
        context.append({k:parent[k] for k in ['id','design','development','hypothesis']})
        frame=root/parent['id']/'development_41_finish.png'
        if frame.exists() and len(images)<2:
            images.append(dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(frame.read_bytes()).decode(),detail='low'))
    text=("Propose a useful structural or dimensional edit to one parent for a physical Override Pin acquisition/transport/goal-placement task. "
          "Use primitive component edits, not an entire robot template. At least explore roller/guide alternatives to a pinching claw when pickup fails. "
          "For replace edits, the replacement part id must exactly equal target. For fixed passive parts, use limits [0,0]. "
          "Allowed catalog: beam, tool_mount, finger, roller, guide. Offsets/sizes in meters. Parent order must precede child. "
          "chassis origin exposes only origin; beams expose tip at [length,0,0]; tool_mount exposes tool at [.03,0,0]. "
          "Roles lift,wrist and either grip_left/right or roller_left/right are required; role names are unique. "
          "Roller axes [1,0,0], x=size length,y=z=diameter; paired rollers inward edges should grip the ~45 mm pin upper taper. "
          "Axis vectors must be unit. Joint kind fixed requires passive role. Total declared motors <=88W, drive44W. "
          "Part dimensions 0.005 to 0.32 m; mounting offsets <=.20m. "
          "Root lift mount [-.0508,0,.14] and arm length .22..30m are known to fit initial envelope. "
          "Physics, benchmark and legal catalog cannot be edited. Unsupported generators go in extension_request, quarantined without execution. "
          "Keep extension_request null for supported edits. Expected benefit is only a hypothesis. "
          "Privileged-state controller gets the same bounded grip-closure tuning budget per design. "
          "Pin height .165 m, tip radius .0178, central flange radius .04015; goal opening radius .03005, rim z .0825. "
          "First task starts upright Pin on floor; close/acquire, lift, drive, lower, release. No hidden attachments.\nParents:\n"+json.dumps(context))
    inputs=[dict(role='user',content=[dict(type='input_text',text=text),*images])]
    for attempt in range(2):
        if len(calls)>=limits['api_calls']:raise RuntimeError('API call budget exhausted')
        record=dict(model=model,reasoning=reasoning,max_output_tokens=limits['output_tokens'],parent_ids=[p['id'] for p in parents],attempt=attempt+1,prompt=text,image_count=len(images))
        calls.append(record);save(path,calls)
        try:
            response=client.responses.parse(model=model,reasoning={'effort':reasoning},input=inputs,text_format=Proposal,max_output_tokens=limits['output_tokens'],store=False)
            record.update(status=response.status,usage=response.usage.model_dump() if response.usage else None,output=response.output_text,response_id=response.id)
            p=response.output_parsed
            if p is None:raise ValueError('refused/incomplete proposal')
            parent=next((x for x in parents if x['id']==p.parent_id),None)
            if parent is None:raise ValueError('unknown parent id')
            robot=apply_edits(Robot.model_validate(parent['design']),p)
            m=mujoco.MjModel.from_xml_string(compile_robot(robot));construction_check(m,initial_data(m,robot),robot)
            record['validated']=True;save(path,calls);return p
        except ValueError as e:
            record['validation_error']=str(e)[:1500];save(path,calls)
            inputs.append(dict(role='user',content='Repair invalid proposal: '+str(e)[:1500]))
        except Exception as e:
            record.update(error_type=type(e).__name__,http_status=getattr(e,'status_code',None));save(path,calls)
            raise RuntimeError(f'API failure {type(e).__name__}, HTTP {getattr(e,"status_code",None)}') from None
    raise ValueError('proposal failed bounded repair')


def report(root,state):
    rows=[];playbacks=[]
    for c in state['candidates']:
        if c.get('validation_error'):
            rows.append(f"<section><h2>{html.escape(c['id'])}: rejected</h2><pre>{html.escape(c['validation_error'])}</pre></section>");continue
        results=c.get('final') or c.get('development') or []
        n=sum(t['success'] for t in results);mean=np.mean([t['time_s'] if t['success'] else state['limits']['duration'] for t in results]) if results else 0
        failures=sorted(set(f for t in results for f in t['failures']))
        images=''.join(f'<a href="{c["id"]}/{p.name}"><img src="{c["id"]}/{p.name}" alt="Recorded physical trial"></a>' for p in sorted((root/c['id']).glob('*_finish.png'))[:2])
        gifs=''.join(f'<p><a href="{c["id"]}/{p.name}">Animated replay: {p.stem}</a></p>' for p in sorted((root/c['id']).glob('*.gif'))[:2])
        hypothesis_outcome='inconclusive'
        parent=next((x for x in state['candidates'] if x['id']==c['parent']),None)
        if parent and results and parent.get('development'):
            base=parent.get('final') if c.get('final') and parent.get('final') else parent['development']
            comparison=c['final'] if c.get('final') and parent.get('final') else c['development']
            a=rank(base);b=rank(comparison);hypothesis_outcome='improvement' if b<a else 'regression' if b>a else 'inconclusive'
        rows.append(f'''<section><h2>{c['id']} · {html.escape(c['architecture'])}</h2><p>Parent: {c['parent']} · {html.escape(c['source'])}</p>
        <p>{html.escape(c['hypothesis'])}</p><p>Hypothesis comparison: {hypothesis_outcome} (ranking on shared split; not statistical significance)</p>
        <p>{n}/{len(results)} successes · mean time with timeout {mean:.2f} s · {'held-out' if c.get('final') else 'development'}</p>
        <p>Failures: {html.escape(', '.join(failures) or 'none')}</p><p><a href="{c['id']}/design.json">Assembly JSON</a> · <a href="{c['id']}/candidate.json">All metrics and edits</a></p>{images}{gifs}</section>''')
        for p in sorted((root/c['id']).glob('development_*.npz'))[:1]:
            t=np.load(p);playbacks.append(dict(label=c['id']+'/'+p.stem,time=t['time'].tolist(),state=t['state'].tolist()))
    payload=json.dumps(playbacks).replace('<','\\u003c')
    text='''<!doctype html><meta charset="utf-8"><title>Override mechanism search</title><style>body{background:#101b25;color:#e7edf4;font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:30px}section{background:#1d2c3a;padding:22px;margin:20px 0;border-radius:12px}a{color:#8cdfed}img{width:45%}canvas{background:#273846;width:100%}input{width:75%}pre{white-space:pre-wrap}</style><h1>Override mechanism experiments</h1>'''
    text+=f"<p>{html.escape(state['mode'])} · {html.escape(state['status'])}</p><p>{html.escape(state.get('blocker',''))}</p><p>Simulation benchmarks based on official field resources. Reconstructed contact geometry; privileged-state controller; partial rules only. No claim of competition legality or hardware performance.</p>"
    text+='''<section><h2>Physical trial inspection</h2><select id="trial"></select><p>Scrub recorded time. Blue square: robot root; yellow circle: object; green cross: grasp point. Goal at x=0.8 m.</p><canvas id="c" width="1000" height="360"></canvas><input id="t" type="range" min="0" max="32" value="0" step=".05"><output id="clock"></output></section>'''
    text+=''.join(rows)+f'<script>const trials={payload};'+'''const sel=document.getElementById('trial'),slider=document.getElementById('t'),ctx=document.getElementById('c').getContext('2d');trials.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=t.label;sel.appendChild(o)});function draw(){ctx.clearRect(0,0,1000,360);const r=trials[Number(sel.value)];if(!r)return;let i=0;while(i+1<r.time.length&&r.time[i+1]<=Number(slider.value))i++;const s=r.state[i];function p(x,y){return[100+x*800,180-y*600]}ctx.strokeStyle='#e54a61';ctx.beginPath();ctx.arc(...p(.8,0),.03005*800,0,7);ctx.stroke();ctx.fillStyle='#7cabed';let q=p(s[3],s[4]);ctx.fillRect(q[0]-15,q[1]-15,30,30);ctx.fillStyle='#f4d162';ctx.beginPath();ctx.arc(...p(s[0],s[1]),8,0,7);ctx.fill();ctx.fillStyle='#7df2bd';q=p(s[6],s[7]);ctx.fillRect(q[0]-3,q[1]-3,6,6);document.getElementById('clock').textContent=r.time[i].toFixed(2)+' s; object z='+s[2].toFixed(3)+' m'}sel.onchange=draw;slider.oninput=draw;draw();</script>'''
    (root/'report.html').write_text(text)


def search(output='mechanism_runs',ai=False,resume=None,generations=3,candidates=3,trials=2,tuning=2,duration=32,api_calls=12,output_tokens=3000,wall_seconds=600,model='gpt-5.6-sol',reasoning='low',render=True):
    start=time.monotonic()
    if resume:
        root=Path(resume);state=json.loads((root/'state.json').read_text())
        if state['code_hashes']!=hashcode():raise ValueError('source changed since checkpoint; refuse mixed-code resume')
        if state['resource_hash']!=hashlib.sha256(Path('resources/override_v2/manual.pdf').read_bytes()).hexdigest():raise ValueError('pinned manual changed')
        ai=state['mode']=='AI-generated proposals';model=state['model'];reasoning=state['reasoning']
    else:
        root=Path(output)/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+('ai' if ai else 'offline')+'_'+uuid.uuid4().hex[:5]);root.mkdir(parents=True)
        state=dict(mode='AI-generated proposals' if ai else 'OFFLINE handwritten development proposals; not AI-generated',status='running',model=model,reasoning=reasoning,
                   limits=dict(generations=generations,candidates=candidates,trials=trials,tuning=tuning,duration=duration,api_calls=api_calls,output_tokens=output_tokens,wall_seconds=wall_seconds),
                   candidates=[],code_hashes=hashcode(),resource_hash=hashlib.sha256(Path('resources/override_v2/manual.pdf').read_bytes()).hexdigest(),wall_used_s=0,
                   seeds=dict(training=TRAIN,development=DEVELOPMENT,final=FINAL),versions={p:importlib.metadata.version(p) for p in ['mujoco','numpy','openai','pydantic']})
        shutil.copytree('resources/override_v2',root/'resources')
        shutil.copytree(Path(__file__).parents[1],root/'source',ignore=shutil.ignore_patterns('__pycache__'))
        save(root/'assembly.schema.json',Robot.model_json_schema());save(root/'proposal.schema.json',Proposal.model_json_schema())
    state['status']='running'
    limits=state['limits'];elapsed_before=state['wall_used_s']
    def checkpoint():
        state['wall_used_s']=elapsed_before+time.monotonic()-start
        save(root/'state.json',state);report(root,state)
    def budget():
        if elapsed_before+time.monotonic()-start>limits['wall_seconds']:raise TimeoutError('wall-clock budget exhausted; checkpoint preserved')
    def evaluate(c,split,seeds,closure,task='pin'):
        results=[];directory=root/c['id'];robot=Robot.model_validate(c['design'])
        for s in seeds:
            budget();stem=f'{split}_{s}';path=directory/(stem+'.json')
            if path.exists():r=json.loads(path.read_text())
            else:
                r,t,xml=run(robot,s,task,duration=limits['duration'],closure=closure)
                save(path,r);(directory/(stem+'.xml')).write_text(xml);np.savez_compressed(directory/(stem+'.npz'),**t)
            results.append(r)
        return results
    def finish_candidate(c):
        directory=root/c['id'];directory.mkdir(exist_ok=True);save(directory/'design.json',c['design'])
        if not c.get('development'):
            scores=[]
            # Exactly the same bounded tuning schedule for each candidate.
            for i in range(limits['tuning']):
                closure=[.58,.66,.72][i%3]
                rs=evaluate(c,f'train_{i}',[TRAIN[i%len(TRAIN)]],closure)
                scores.append((rank(rs),closure))
            c['controller_closure']=min(scores)[1] if scores else .62
            c['development']=evaluate(c,'development',DEVELOPMENT[:limits['trials']],c['controller_closure'])
            if render:
                try:
                    p=subprocess.run([sys.executable,'-m','chassis_lab.override.visual',str(directory/'development_41.xml')],capture_output=True,timeout=25)
                    c['rendered']=p.returncode==0
                except Exception:c['rendered']=False
        save(directory/'candidate.json',c);checkpoint()
        print(c['id'],c['architecture'],sum(t['success'] for t in c['development']), '/',len(c['development']),flush=True)
    try:
        checkpoint()
        if not state['candidates']:
            robot=seed();state['candidates'].append(dict(id='seed',parent=None,design=robot.model_dump(),architecture=robot.architecture,source='image-inspired handwritten seed',hypothesis='Arm and claw can acquire and place one upright Pin.'))
        finish_candidate(state['candidates'][0])
        for generation in range(limits['generations']):
            # Select champion, distinct architecture and original seed; freeze for generation.
            earlier=[c for c in state['candidates'] if c.get('development') and c.get('generation',-1)<generation]
            ordered=sorted(earlier,key=lambda c:rank(c['development']));parents=[ordered[0]]
            different=next((c for c in ordered if c['architecture']!=ordered[0]['architecture']),None)
            if different:parents.append(different)
            if state['candidates'][0] not in parents:parents.append(state['candidates'][0])
            for index in range(limits['candidates']):
                budget();cid=f'g{generation+1}_c{index+1}'
                existing=next((c for c in state['candidates'] if c['id']==cid),None)
                if existing:
                    if not existing.get('validation_error'):finish_candidate(existing)
                    continue
                if ai:proposal=proposal_ai(parents,model,reasoning,root,limits)
                else:
                    parent=parents[index%len(parents)]
                    proposal=fixture_edit(Robot.model_validate(parent['design']),parent['id'],generation,index)
                parent=next(c for c in parents if c['id']==proposal.parent_id)
                try:
                    robot=apply_edits(Robot.model_validate(parent['design']),proposal)
                    smoke_m=mujoco.MjModel.from_xml_string(compile_robot(robot));smoke_d=initial_data(smoke_m,robot)
                    check=construction_check(smoke_m,smoke_d,robot)
                    for _ in range(50):mujoco.mj_step(smoke_m,smoke_d)
                    if not np.isfinite(smoke_d.qpos).all():raise ValueError('nonfinite smoke test')
                    c=dict(id=cid,generation=generation,parent=parent['id'],design=robot.model_dump(),architecture=robot.architecture,
                           source='OpenAI Responses API' if ai else 'handwritten fixture edit',hypothesis=proposal.hypothesis,proposal=proposal.model_dump(),smoke=check)
                except ValueError as e:
                    c=dict(id=cid,generation=generation,parent=parent['id'],validation_error=str(e),proposal=proposal.model_dump())
                    state['candidates'].append(c);checkpoint();continue
                state['candidates'].append(c);checkpoint();finish_candidate(c)
        ordered=sorted([c for c in state['candidates'] if c.get('development')],key=lambda c:rank(c['development']))
        finalists=[state['candidates'][0]]
        if ordered[0] not in finalists:finalists.append(ordered[0])
        other=next((c for c in ordered if c['architecture']!=ordered[0]['architecture']),None)
        if other and other not in finalists:finalists.append(other)
        for c in finalists:
            c['final']=evaluate(c,'final',FINAL[:limits['trials']],c['controller_closure'])
            c['cup_extension']=evaluate(c,'cup',FINAL[:1],c['controller_closure'],task='cup')
            save(root/c['id']/'candidate.json',c);checkpoint()
        state['status']='complete';state.pop('blocker',None)
    except (TimeoutError,RuntimeError,ValueError) as e:
        state['status']='blocked' if ai else 'incomplete';state['blocker']=str(e)
    finally:checkpoint()
    print('Run:',root.resolve(),'status:',state['status'],flush=True)
    return root
