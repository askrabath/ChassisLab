from typing import Literal
import math
import numpy as np
from pydantic import Field,model_validator
from ..schema import StrictModel

class Joint(StrictModel):
    kind: Literal['fixed','revolute','prismatic']
    axis: list[float]=Field(min_length=3,max_length=3)
    limits: list[float]=Field(min_length=2,max_length=2)
    reduction: Literal[1,3,5]
    role: Literal['passive','lift','wrist','grip_left','grip_right','roller_left','roller_right']

class Part(StrictModel):
    id: str=Field(pattern=r'^[a-z][a-z0-9_]{0,31}$')
    parent: str
    attachment: Literal['origin','tip','tool']
    catalog: Literal['beam','tool_mount','finger','roller','guide']
    offset_m: list[float]=Field(min_length=3,max_length=3)
    size_m: list[float]=Field(min_length=3,max_length=3,description='Full x/y/z dimensions; roller x=length,y=z=diameter')
    joint: Joint

class Robot(StrictModel):
    version: Literal['assembly-1']
    name: str=Field(min_length=1,max_length=80)
    parts: list[Part]=Field(min_length=3,max_length=14)
    @model_validator(mode='after')
    def assembly_checks(self):
        available={'chassis'};roles=[];byid={p.id:p for p in self.parts}
        if len(byid)!=len(self.parts):raise ValueError('duplicate part id')
        for p in self.parts:
            if p.parent not in available:raise ValueError(f'{p.id}: parent must precede child and connect to chassis')
            if p.parent=='chassis' and p.attachment!='origin':raise ValueError('chassis exposes origin only')
            if p.attachment=='tip' and byid[p.parent].catalog!='beam':raise ValueError('tip attachment exists only on beams')
            if p.attachment=='tool' and byid[p.parent].catalog!='tool_mount':raise ValueError('tool attachment exists only on tool_mount')
            if max(abs(x) for x in p.offset_m)>.20:raise ValueError('mount offset exceeds 0.20 m')
            if not all(.005<=x<=.32 for x in p.size_m):raise ValueError('part dimensions must be 5..320 mm')
            if p.joint.limits[0]>=p.joint.limits[1]:raise ValueError('joint limits must increase')
            if p.joint.kind!='fixed' and abs(np.linalg.norm(p.joint.axis)-1)>.001:raise ValueError('joint axis must be unit length')
            if p.joint.kind=='prismatic' and max(abs(x) for x in p.joint.limits)>.12:raise ValueError('slide travel exceeds supported 120 mm')
            if p.joint.kind=='prismatic' and p.joint.role!='passive':raise ValueError('prismatic geometry supported; powered slides require a separately validated linear controller/transmission extension')
            if p.joint.kind=='fixed' and p.joint.role!='passive':raise ValueError('fixed component cannot be actuated')
            if p.catalog=='roller' and abs(p.size_m[1]-p.size_m[2])>1e-8:raise ValueError('roller y and z diameters must match')
            if p.joint.role!='passive':roles.append(p.joint.role)
            available.add(p.id)
        if len(roles)!=len(set(roles)):raise ValueError('each controller role must be unique')
        if not {'lift','wrist'}.issubset(roles):raise ValueError('this controller requires lift and wrist roles')
        if not ({'grip_left','grip_right'}.issubset(roles) or {'roller_left','roller_right'}.issubset(roles)):
            raise ValueError('need an actuated grip pair or roller pair')
        watts=44+sum(5.5 if r.startswith('grip') else 11 for r in roles)
        if watts>88:raise ValueError('declared motors exceed implemented 88 W screen')
        return self
    @property
    def architecture(self):
        return '+'.join(sorted(set(p.catalog for p in self.parts)))

class Edit(StrictModel):
    operation: Literal['add','remove','replace']
    target: str
    part: Part|None

class Proposal(StrictModel):
    parent_id: str
    edits: list[Edit]=Field(min_length=1,max_length=10)
    hypothesis: str=Field(max_length=1000)
    expected_effect: str=Field(max_length=600)
    extension_request: str|None=Field(description='Unsupported generator proposal; quarantined, never executed')


def apply_edits(parent,proposal):
    if proposal.extension_request:raise ValueError('unsupported generator extension quarantined for separate implementation/review')
    parts=[p.model_dump() for p in parent.parts]
    for e in proposal.edits:
        ids=[p['id'] for p in parts]
        if e.operation=='add':
            if e.part is None or e.part.id in ids:raise ValueError('add needs a new part id')
            parts.append(e.part.model_dump())
        else:
            if e.target not in ids:raise ValueError('edit target does not exist')
            i=ids.index(e.target)
            if e.operation=='remove':parts.pop(i)
            else:
                if e.part is None or e.part.id!=e.target:raise ValueError('replacement must retain target id')
                parts[i]=e.part.model_dump()
    return Robot(version='assembly-1',name=parent.name+' revision',parts=parts)


def joint(role='passive',kind='fixed',axis=(0,0,1),limits=(-1,1),reduction=1):
    return Joint(kind=kind,axis=list(axis),limits=list(limits),reduction=reduction,role=role)


def seed():
    return Robot(version='assembly-1',name='Image-inspired arm and claw',parts=[
        Part(id='arm',parent='chassis',attachment='origin',catalog='beam',offset_m=[-.0508,0,.14],size_m=[.27,.04,.0127],joint=joint('lift','revolute',(0,-1,0),(-.45,1.1),5)),
        Part(id='wrist',parent='arm',attachment='tip',catalog='tool_mount',offset_m=[0,0,0],size_m=[.03,.14,.016],joint=joint('wrist','revolute',(0,-1,0),(-1.2,.6),3)),
        Part(id='left_finger',parent='wrist',attachment='tool',catalog='finger',offset_m=[0,.06,0],size_m=[.08,.009,.04],joint=joint('grip_left','revolute',(0,0,1),(-.8,.35))),
        Part(id='right_finger',parent='wrist',attachment='tool',catalog='finger',offset_m=[0,-.06,0],size_m=[.08,.009,.04],joint=joint('grip_right','revolute',(0,0,1),(-.35,.8)))])


def fixture_edit(parent,parent_id,generation,index):
    """Explicit deterministic controls for development; never AI-generated."""
    edits=[]
    if index==0 and generation==0:
        for ident,role,y in [('left_finger','roller_left',.043),('right_finger','roller_right',-.043)]:
            p=Part(id=ident,parent='wrist',attachment='tool',catalog='roller',offset_m=[.065,y,0],size_m=[.055,.042,.042],joint=joint(role,'revolute',(1,0,0),(-10000,10000)))
            edits.append(Edit(operation='replace',target=ident,part=p))
        p=Part(id='back_guide',parent='wrist',attachment='tool',catalog='guide',offset_m=[.015,0,-.015],size_m=[.008,.07,.045],joint=joint())
        edits.append(Edit(operation='add',target='back_guide',part=p))
    else:
        target=parent.parts[0].model_copy(deep=True)
        target.size_m[0]=float(np.clip(target.size_m[0]+(.0127 if index%2 else -.0127),.22,.30))
        edits.append(Edit(operation='replace',target=target.id,part=target))
        if index==2 and 'back_guide' not in [p.id for p in parent.parts]:
            p=Part(id='back_guide',parent='wrist',attachment='tool',catalog='guide',offset_m=[.015,0,-.015],size_m=[.008,.07,.045],joint=joint())
            edits.append(Edit(operation='add',target=p.id,part=p))
    return Proposal(parent_id=parent_id,edits=edits,hypothesis='Handwritten development mutation: test contact acquisition after a structural or reach change.',expected_effect='Measure pickup and placement; no improvement assumed.',extension_request=None)
