"""Download only explicitly pinned public official-resource links."""
from pathlib import Path
from urllib.request import urlopen, Request
import hashlib,json,datetime

SOURCES={
 'manual.pdf':'https://content.vexrobotics.com/docs/2026-2027/override/files/override-2.0.pdf',
 'field_cad.zip':'https://link.vex.com/docs/26-27/v5rc/field-cad',
 'assembly.html':'https://link.vex.com/docs/26-27/v5rc/buildinstructions',
 'manual_complement.html':'https://www.vexrobotics.com/override-manual',
 'game_page.html':'https://www.vexrobotics.com/v5/competition/vrc-current-game',
}
root=Path('resources/override_v2');root.mkdir(parents=True,exist_ok=True)
manifest={}
for name,url in SOURCES.items():
 try:
  with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:
   data=r.read();final=r.url;mime=r.headers.get('Content-Type')
  (root/name).write_bytes(data)
  manifest[name]=dict(source_url=url,resolved_url=final,content_type=mime,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  print(name,len(data),final,flush=True)
 except Exception as e: manifest[name]=dict(source_url=url,error=type(e).__name__,status=getattr(e,'code',None));print(name,type(e).__name__,getattr(e,'code',None),flush=True)
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
