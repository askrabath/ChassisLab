from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os,json,datetime,hashlib

had_environment_key=bool(os.environ.get('OPENAI_API_KEY'))
loaded_local=load_dotenv('.env.local',override=False)
loaded_env=load_dotenv('.env',override=False)
key=os.environ.get('OPENAI_API_KEY','')
source='process environment' if had_environment_key else '.env.local' if loaded_local else '.env' if loaded_env else 'missing'
r={
    'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model':os.environ.get('CHASSIS_LAB_MODEL','gpt-5.6-sol'),
    'credential':{
        'source':source,
        'present':bool(key),
        'project_key_format':key.startswith('sk-proj-'),
        'length':len(key) if key else None,
        # Safe correlation identifier: this cannot be used as the API key.
        'sha256_prefix':hashlib.sha256(key.encode()).hexdigest()[:12] if key else None,
    },
}
try:
 response=OpenAI(max_retries=0,timeout=30).responses.create(model=r['model'],input='Reply OK for a connectivity check.',max_output_tokens=32)
 r.update(status=response.status,usage=response.usage.model_dump() if response.usage else None)
except Exception as e:r.update(error_type=type(e).__name__,http_status=getattr(e,'status_code',None))
Path('resources/override_v2/api_preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
