import html
import json
from pathlib import Path
import numpy as np
from .physics import WAYPOINTS, OBSTACLES


def write_report(root, manifest, candidates, verdict):
    root=Path(root)
    sections=[]; tracks=[]
    for item in candidates:
        index=item['id']; name=f'candidate_{index}'
        proposal=item['proposal']
        rows=[]
        for split,agg in item['aggregates'].items():
            rows.append(f"<tr><td>{split}</td><td>{agg['clean_completions']}/{agg['trials']}</td><td>{agg['mean_time_with_timeout_s']:.3f}</td><td>{agg['mean_tracking_error_m']:.4f}</td><td>{agg['collision_episodes']}</td><td>{agg['mean_progress']:.1%}</td></tr>")
        media=''
        if (root/name/'robot.png').exists():
            media=f'<img src="{name}/robot.png" alt="MuJoCo chassis rendering"><img src="{name}/slalom.png" alt="MuJoCo course rendering">'
        sections.append(f'''<section><h2>{'Initial' if index==0 else 'Revision'} · {html.escape(item['source'])}</h2>
        <p>{html.escape(proposal['rationale'])}</p><p><b>Predicted benefit:</b> {html.escape(proposal['predicted_benefit'])}</p>
        <p><b>Hypothesis:</b> {html.escape(proposal['hypothesis'])}</p><p>Parent: {item['parent']} · Mass: {item['mass_kg']:.3f} kg</p>
        <table><thead><tr><th>Split</th><th>Clean finishes</th><th>Mean time / timeout (s)</th><th>RMS (m)</th><th>Collisions</th><th>Progress</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
        <p><a href="{name}/design.json">Design JSON</a> · <a href="{name}/model.xml">MJCF</a> · <a href="{name}/results.json">Trial metrics</a></p>
        {media}<details><summary>Specification</summary><pre>{html.escape(json.dumps(proposal['design'],indent=2))}</pre></details></section>''')
        path=root/name/'final_101.npz'
        if path.exists():
            tracks.append(dict(name='Initial' if index==0 else 'Revision',points=np.load(path)['trace'].tolist(),design=proposal['design']))
    payload=json.dumps(dict(tracks=tracks,waypoints=WAYPOINTS.tolist(),obstacles=OBSTACLES)).replace('<','\\u003c')
    failure=f"<p class='warning'>{html.escape(manifest.get('failure',''))}</p>" if manifest.get('failure') else ''
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Chassis Lab comparison</title><style>
    body{font:16px/1.5 system-ui;background:#111821;color:#e8edf5;max-width:1120px;margin:auto;padding:32px}h1{font-size:36px}h2{font-size:23px}p{max-width:95ch}section{background:#1b2633;padding:24px;margin:24px 0;border-radius:12px}a{color:#78dded}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #445366;padding:10px}img{width:48%;border-radius:8px}pre{white-space:pre-wrap}canvas{width:100%;background:#16212a;border-radius:10px}input{width:80%}.warning{color:#ffb389}.tag{color:#73e5c7}</style>
    <h1>Chassis Lab</h1>'''+f'''<p class="tag">{html.escape(manifest['mode'])} · {html.escape(verdict.upper())}</p>{failure}
    <p>A synthetic engineering benchmark using wheel-actuated MuJoCo physics. Approximate VEX-inspired components; no official legality or hardware-performance claim.</p>
    <p>The verdict uses three paired held-out trials. Ranking prioritizes clean finishes, then completion, stability, collisions, progress, time and tracking. This small experiment does not establish statistical significance.</p>
    <section><h2>Recorded course playback · held-out seed 101</h2><p>Blue: initial. Green: revision. Scrub elapsed simulation time; full trails show the measured paths. This is recorded-state visualization.</p>
    <canvas id="course" width="1100" height="510"></canvas><label>Time <input id="time" type="range" min="0" max="45" step="0.02" value="0"></label><output id="clock">0 s</output></section>
    {''.join(sections)}<section><h2>Experiment record</h2><p><a href="manifest.json">Configuration, seeds, versions and hashes</a> · <a href="assumptions.json">Modeling assumptions</a> · <a href="comparison.json">Aggregate comparison</a></p><p>AI calls, outputs, repairs and available token usage are in ai_calls.json for AI runs. Offline proposals are handwritten sample fixtures, not AI-generated.</p></section>
    <script>const data={payload};'''+'''
    const c=document.getElementById('course'),ctx=c.getContext('2d'),slider=document.getElementById('time');
    function point(x,y){return [70+(x+0.5)*142,255-y*130]}
    function draw(){ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#596a7d';ctx.strokeRect(...point(-0.75,1.75),7.2*142,3.5*130);
    ctx.setLineDash([6,6]);ctx.beginPath();data.waypoints.forEach((p,i)=>{let q=point(...p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.setLineDash([]);
    data.obstacles.forEach(p=>{ctx.fillStyle='#ed853d';ctx.beginPath();ctx.arc(...point(...p),.14*136,0,Math.PI*2);ctx.fill()});
    ctx.strokeStyle='#66dba8';ctx.strokeRect(...point(5.35,.3),.6*142,.6*130);
    data.tracks.forEach((track,i)=>{ctx.strokeStyle=i?'#73e5c7':'#79a9ff';ctx.beginPath();track.points.forEach((p,j)=>{let q=point(p[1],p[2]);j?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
    let p=track.points[0];for(let q of track.points){if(q[0]>Number(slider.value))break;p=q}if(!p)return;
    let q=point(p[1],p[2]);ctx.save();ctx.translate(...q);ctx.rotate(-p[3]);ctx.fillStyle=i?'#73e5c7':'#79a9ff';let l=track.design.frame_length_m*142,w=track.design.track_width_m*130;ctx.fillRect(-l/2,-w/2,l,w);ctx.strokeStyle='#101923';ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(l/2,0);ctx.stroke();ctx.restore()});
    document.getElementById('clock').textContent=Number(slider.value).toFixed(2)+' s'}slider.oninput=draw;draw();</script></html>'''
    (root/'report.html').write_text(text)
