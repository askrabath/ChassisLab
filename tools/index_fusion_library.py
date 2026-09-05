"""Index downloaded CAD without claiming filename matching proves legality."""
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile
from .official_catalog import ROOT

def build():
    folder=ROOT/'data/official_catalog';archive=folder/'community_parts.zip'
    thumbs=folder/'thumbnails';thumbs.mkdir(exist_ok=True)
    legal=json.loads((folder/'catalog.json').read_text())
    items=[]
    with zipfile.ZipFile(archive) as z:
        for n in sorted(z.namelist()):
            if not n.endswith('.f3d'):continue
            key=hashlib.sha256(n.encode()).hexdigest()[:16];preview=None
            raw=z.read(n)
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as f:
                    candidates=[s for s in f.namelist() if s.endswith('/Previews/small.png')]
                    if candidates:
                        preview=f'thumbnails/{key}.png';(folder/preview).write_bytes(f.read(candidates[0]))
            except zipfile.BadZipFile:pass
            items.append(dict(id=key,name=Path(n).stem,archive_member=n,format='Fusion 360 F3D',sha256=hashlib.sha256(raw).hexdigest(),thumbnail=preview,
                sku=None,legal_status='UNRESOLVED: cross-reference official SKU before use',
                mujoco_geometry_status='requires STEP/mesh export from Fusion',verified_connections=None))
    manifest=dict(source='https://github.com/VEX-CAD/VEX-CAD-Fusion-360-Library',
        source_commit=json.loads((folder/'community_tree.json').read_text()).get('sha'),
        archive='community_parts.zip',archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),
        files=len(items),parts=items,authority='Community CAD, not competition rules authority')
    (folder/'fusion_catalog.json').write_text(json.dumps(manifest,indent=2))
    legal['community_cad']=dict(manifest='fusion_catalog.json',downloaded_f3d_files=len(items),step_conversion_complete=False,sku_cross_reference_complete=False)
    (folder/'catalog.json').write_text(json.dumps(legal,indent=2))
    cards=[]
    for p in items:
        image=f'<img loading="lazy" src="{p["thumbnail"]}">' if p['thumbnail'] else '<div>No embedded thumbnail</div>'
        cards.append('<article>'+image+'<b>'+html.escape(p['name'])+'</b><small>'+html.escape(p['archive_member'].split('/',1)[1])+'</small></article>')
    (folder/'cad_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Downloaded CAD library</title><style>body{font:16px system-ui;background:#14202c;color:white;margin:30px}input{padding:12px;width:70%}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;margin-top:20px}article{background:#26394b;padding:12px}img{width:180px;height:140px;object-fit:contain}b,small{display:block;overflow-wrap:anywhere}a{color:#8ee}</style><h1>'+str(len(items))+' downloaded Fusion CAD files</h1><p>Washers, collars, bearings, shafts, structure and electronics. Images are embedded CAD thumbnails, not AI illustrations. Files are preserved inside community_parts.zip. Individual SKU legality mapping and STEP conversion are incomplete.</p><a href="index.html">Official legal-parts list</a> · <a href="fusion_catalog.json">CAD inventory JSON</a><p><input placeholder="Search parts, e.g. washer, collar, bearing" oninput="document.querySelectorAll(\'article\').forEach(e=>e.hidden=!e.innerText.toLowerCase().includes(this.value.toLowerCase()))"></p><main>'+''.join(cards)+'</main>')
    index=folder/'index.html';text=index.read_text();text=text.replace('<h1>','<p><a href="cad_gallery.html">Browse '+str(len(items))+' downloaded CAD files and thumbnails</a></p><h1>',1);index.write_text(text)
    print(json.dumps(dict(downloaded=len(items),thumbnails=sum(bool(p['thumbnail']) for p in items))))

if __name__=='__main__':build()
