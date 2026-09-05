"""Versioned legal-parts inventory from the official R16-linked publication.

Legal listing is not CAD availability and does not certify an assembled robot.
"""
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://docs.google.com/spreadsheets/d/e/2PACX-1vQxumg3GopriUY8RF5cBCl_KomUCh_OeFnYosLip3rnEFuFYWdneuZUBEyODj52kCqCQCtvD3J2P4EQ/pub?output=csv&gid=1118739975'

def build():
    folder=ROOT/'data/official_catalog';raw=folder/'legal_rows.csv'
    rows=list(csv.DictReader(raw.open()));parts={};source_tree=json.loads((ROOT/'data/cad_assets/clawbot/assembly_tree.json').read_text())
    for line,row in enumerate(rows,2):
        sku=row['SKU'].strip();name=' '.join(row['Name'].split())
        if not sku:continue
        if sku not in parts:parts[sku]=dict(sku=sku,name=name,category=row['Part Type'].strip().replace('Fasterner','Fastener') or 'Uncategorized',
            source_rows=[],legal_status='listed_in_official_R16_parts_list',legal_source=SOURCE,
            cad_status='not_downloaded',cad=[],connection_points=None,mass_kg=None,
            assembly_legality='not_evaluated',restrictions='Subject to current manual, official Q&A, modification, motor and assembly rules.')
        parts[sku]['source_rows'].append(line)
    for key,d in source_tree['definitions'].items():
        match=re.search(r'\b27[56]-\d{4}(?:-\d{3})?\b',d['name'])
        if not match or match[0] not in parts:continue
        p=parts[match[0]];p['cad_status']='source_clawbot_configuration_available'
        p['cad'].append(dict(definition=key,name=d['name'],high_mesh=d['mesh'],low_mesh=d['mesh'].replace('.obj','_lod.obj'),
            source_step='data/cad_assets/clawbot/canonical.step',configuration='as supplied; may be cut/configured, not stock full length',bounds_mm=d['bounds_mm']))
    catalog=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),source=SOURCE,authority_link='https://link.vex.com/v5rc-legal-parts',
        manual='https://www.vexrobotics.com/override-manual',raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        source_rows=len(rows),unique_skus=len(parts),parts=parts,
        coverage=dict(legal_inventory='all nonempty-SKU rows in downloaded official sheet',cad_skus=sum(bool(p['cad']) for p in parts.values()),
            complete_cad_library=False,verified_connection_catalog=False),
        limitations=['A legal part does not make every mechanism or modification legal.',
            'Do not infer eligibility from SKU prefixes: the official list includes selected parts from other VEX lines.',
            'No guessed CAD geometry or connection locations are substituted for missing data.'])
    (folder/'catalog.json').write_text(json.dumps(catalog,indent=2))
    listing=''.join('<tr><td>'+html.escape(p['sku'])+'</td><td>'+html.escape(p['name'])+'</td><td>'+html.escape(p['category'])+'</td><td>'+html.escape(p['cad_status'])+'</td></tr>' for p in parts.values())
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Official V5 parts inventory</title><style>body{font:15px system-ui;margin:30px}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}input{font:inherit;padding:10px;width:70%}</style><h1>Official V5 legal-parts inventory</h1><p>'+str(len(parts))+' unique SKUs; '+str(catalog['coverage']['cad_skus'])+' have configurations from supplied CAD. This is NOT a complete downloaded CAD library or a robot legality certification.</p><p><a href="https://link.vex.com/v5rc-legal-parts">Official source</a> · <a href="catalog.json">AI catalog JSON</a></p><input placeholder="Filter by part, SKU, category" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase()))"><table><thead><tr><th>SKU</th><th>Name</th><th>Category</th><th>CAD coverage</th></tr></thead><tbody>'+listing+'</tbody></table>')
    print(json.dumps(dict(source_rows=len(rows),unique_skus=len(parts),coverage=catalog['coverage']),indent=2))
    return catalog

if __name__=='__main__':build()
