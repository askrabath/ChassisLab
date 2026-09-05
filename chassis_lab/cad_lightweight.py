"""Visual-only LOD and explicit local field bounds; preserve retained contacts."""
import json
from pathlib import Path
import numpy as np
from .cad_scene import ROOT,tree,instance_pose
from .physics import element

REGION=((-1.5,.55),(-1.5,-.3))

def optimize(root,field_transform,lightweight=True,crop=True):
    asset=root.find('asset');world=root.find('worldbody');field=tree('override_field')
    retained=set();removed=[]
    for item in field['instances']:
        definition=field['definitions'][item['definition']]
        lo,hi=np.array(definition['bounds_mm'])/1000
        corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        points=(corners@instance_pose(item,field_transform).T)[:,:3]
        intersects=all(points[:,axis].max()>=limits[0] and points[:,axis].min()<=limits[1] for axis,limits in enumerate(REGION))
        if not crop or intersects:retained.add(item['id'])
    for geom in list(world.findall('geom')):
        name=geom.get('name','')
        if (name.startswith('field_i') or name.startswith('contact_i')) and name.split('_')[-1] not in retained:
            removed.append(name);world.remove(geom)
    if crop:
        # A local floor surface is a display boundary, not an invented wall.
        for geom in list(world.findall('geom')):
            if geom.get('name','').startswith('field_'):
                key=geom.get('mesh','').removeprefix('field_')
                if key=='p00032' or 'Tape' in field['definitions'].get(key,{}).get('name',''):
                    world.remove(geom)
        center=[sum(x)/2 for x in REGION];half=[(x[1]-x[0])/2 for x in REGION]
        element(world,'geom',name='local_floor_visual',type='box',pos=f'{center[0]} {center[1]} -.009',size=f'{half[0]} {half[1]} .008',mass=0,contype=0,conaffinity=0,rgba='.3 .32 .34 1')
        ground=world.find("geom[@name='display_ground']")
        if ground is not None:world.remove(ground)
    if lightweight:
        robot=tree('clawbot');definitions=robot['definitions']
        for parent in root.iter():
            for geom in list(parent.findall('geom')):
                if geom.get('group')!='1' or geom.get('type')!='mesh':continue
                old=geom.get('mesh');mesh=asset.find(f"mesh[@name='{old}']")
                # Preserve visible goal/cup openings: a convex fallback would
                # make these look sealed. Only a few remain in the local scene.
                if old in {'field_p00034','field_p00036','field_p00051','field_p00062','field_p00063'}:continue
                # Keep washers, bearings and collars visible. Reduce their
                # display meshes without removing their assembly instances.
                new=old+'_visual_lod';path=Path(mesh.get('file'));lod=path.with_name(path.stem+'_lod.obj')
                if not lod.exists():raise FileNotFoundError('Run .venv-cad/bin/python -m tools.cad_lod to build visual meshes')
                if asset.find(f"mesh[@name='{new}']") is None:
                    element(asset,'mesh',name=new,file=str(lod),scale=mesh.get('scale'),inertia='shell')
                geom.set('mesh',new)
        visual=root.find('visual');quality=element(visual,'quality',shadowsize=512,offsamples=0)
        # Viewer and exported images share cheaper visual defaults.
        for light in world.findall('light'):light.set('castshadow','false')
        visual.find('global').set('offwidth','800');visual.find('global').set('offheight','600')
    used={g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') not in used:asset.remove(mesh)
    return dict(lightweight_visuals=lightweight,local_field=crop,region_xy_m=REGION if crop else None,
                removed_field_geoms=len(removed),retained_field_instances=len(retained),
                physics_note='Retained contact meshes, mass, joints, timestep and motor limits unchanged; no artificial crop boundary walls.')
