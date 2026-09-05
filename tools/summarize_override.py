"""Derive aggregate records from immutable per-trial results."""
import json
import sys
from pathlib import Path
from chassis_lab.override.sim import rank
from chassis_lab.override.search import report

root=Path(sys.argv[1]);state=json.loads((root/'state.json').read_text());summary=[]
for c in state['candidates']:
    if not c.get('development'):continue
    row={k:c[k] for k in ['id','parent','architecture','source']}
    for split in ['development','final','cup_extension']:
        trials=c.get(split,[])
        if not trials:continue
        times=[t['time_s'] for t in trials if t['success']]
        row[split]=dict(trials=len(trials),successes=len(times),success_rate=len(times)/len(trials),mean_success_time_s=sum(times)/len(times) if times else None,ranking=rank(trials),failures=sorted(set(f for t in trials for f in t['failures'])),collision_episodes=sum(t['collision_episodes'] for t in trials),unstable_trials=sum(t['unstable'] for t in trials))
    (root/c['id']/'aggregate.json').write_text(json.dumps(row,indent=2));summary.append(row)
(root/'summary.json').write_text(json.dumps(summary,indent=2));report(root,state)
print(json.dumps(summary,indent=2))
