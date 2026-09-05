import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .schema import Design, Proposal, sample_design
from .physics import compile_mjcf, DEV_SEEDS, FINAL_SEEDS, DT
from .benchmark import simulate, aggregate, compare, TIME_LIMIT
from .assumptions import ASSUMPTIONS
from .report import write_report


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def run_experiment(ai=False,model='gpt-5.6-sol',output='runs',render=True):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(output)/f'{stamp}_{"ai" if ai else "offline"}_{uuid.uuid4().hex[:6]}'
    root.mkdir(parents=True,exist_ok=False)
    save=lambda name,obj:save_json(root/name,obj)
    versions={x:importlib.metadata.version(x) for x in ['mujoco','numpy','pydantic','openai','pillow']}
    manifest=dict(mode='AI proposals' if ai else 'OFFLINE handwritten sample designs (not AI-generated)',
        created_utc=stamp,model=model if ai else None,development_seeds=DEV_SEEDS,final_seeds=FINAL_SEEDS,
        versions=versions,python=sys.version,platform=platform.platform(),timestep_s=DT,timeout_s=TIME_LIMIT,
        ai_limits=dict(max_calls=4,max_output_tokens_per_call=2500,max_repairs_per_proposal=1,generations=2),
        source_hashes={},status='running',rendering={})
    snapshot=root/'source';snapshot.mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        manifest['source_hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path,snapshot/path.name)
    save('manifest.json',manifest);save('assumptions.json',ASSUMPTIONS)
    save('design.schema.json',Design.model_json_schema()); save('proposal.schema.json',Proposal.model_json_schema())
    candidates=[];verdict='inconclusive'

    def evaluate(item,split,seeds):
        directory=root/f"candidate_{item['id']}"
        design=Design.model_validate(item['proposal']['design'])
        trials=[]
        for seed in seeds:
            trial,trajectory,_=simulate(design,seed)
            trials.append(trial)
            save_json(directory/f'{split}_{seed}.json',trial.model_dump())
            (directory/f'{split}_{seed}.xml').write_text(compile_mjcf(design,trial.friction))
            np.savez_compressed(directory/f'{split}_{seed}.npz',**trajectory)
            print(f"candidate {item['id']} {split} seed {seed}: complete={trial.completed} time={trial.elapsed_s:.3f}s collisions={trial.collision_episodes}",flush=True)
        item['aggregates'][split]=aggregate(trials)
        item['trials'][split]=[t.model_dump() for t in trials]
        save_json(directory/'results.json',dict(aggregates=item['aggregates'],trials=item['trials']))

    try:
        designer=None
        if ai:
            from .ai import Designer
            # Seeds are recorded locally but withheld from the AI input.
            context={k:v for k,v in ASSUMPTIONS.items() if k!='randomization'}
            context['variation']='Paired trials with +/-0.025 m initial x/y, +/-0.04 rad heading, friction 0.55..0.80. Separate held-out trials.'
            designer=Designer(model,save,context)
        for index in range(2):
            if ai:
                parent=Proposal.model_validate(candidates[0]['proposal']) if index else None
                feedback=dict(aggregate=candidates[0]['aggregates']['development'],trials=candidates[0]['trials']['development']) if index else None
                if feedback:
                    feedback['trials']=[{k:v for k,v in t.items() if k not in ('seed','initial_pose')} for t in feedback['trials']]
                proposal=designer.propose(parent,feedback)
            else:
                proposal=Proposal(design=sample_design(bool(index)),
                    rationale='Handwritten sample: shorter wheelbase to reduce skid-steer scrub.' if index else 'Handwritten sample: centered battery and conventional long wheelbase.',
                    predicted_benefit='Faster cornering with less wheel scrub.' if index else 'Stable four-wheel drive through the complete course.',
                    hypothesis='At least 5% lower mean completion time with no loss of clean completions.' if index else 'Complete all three development trials without collision.')
            directory=root/f'candidate_{index}';directory.mkdir()
            item=dict(id=index,parent=0 if index else None,source='OpenAI Responses API' if ai else 'handwritten fixture',
                      proposal=proposal.model_dump(),mass_kg=proposal.design.total_mass_kg,aggregates={},trials={})
            candidates.append(item)
            save_json(directory/'proposal.json',dict(parent=item['parent'],source=item['source'],**proposal.model_dump()))
            save_json(directory/'design.json',proposal.design.model_dump())
            (directory/'model.xml').write_text(compile_mjcf(proposal.design))
            evaluate(item,'development',DEV_SEEDS)
            save('candidates.json',candidates)
        # Freeze both proposals before final trials; final outcomes never feed revision.
        for item in candidates:
            evaluate(item,'final',FINAL_SEEDS)
        verdict=compare(candidates[0]['aggregates']['final'],candidates[1]['aggregates']['final'])
        manifest['status']='complete'
    except Exception as exc:
        from .ai import ProposalFailure
        manifest['status']='failed'
        manifest['failure']=str(exc) if isinstance(exc,ProposalFailure) else type(exc).__name__
        print('Experiment stopped:',manifest['failure'],flush=True)
    finally:
        save('candidates.json',candidates)
        if render:
            for item in candidates:
                if 'final' not in item['aggregates']: continue
                directory=root/f"candidate_{item['id']}"
                try:
                    process=subprocess.run([sys.executable,'-m','chassis_lab.render',str(directory),'final','101'],
                                           capture_output=True,timeout=30)
                    manifest['rendering'][str(item['id'])]=dict(success=process.returncode==0,returncode=process.returncode)
                except Exception as exc:
                    manifest['rendering'][str(item['id'])]=dict(success=False,error_type=type(exc).__name__)
        save('manifest.json',manifest)
        save('comparison.json',dict(verdict=verdict,candidates=[dict(id=i['id'],parent=i['parent'],aggregates=i['aggregates']) for i in candidates]))
        write_report(root,manifest,candidates,verdict)
    print(f'{verdict.upper()} · report: {root.resolve() / "report.html"}',flush=True)
    return root,manifest['status']=='complete'
