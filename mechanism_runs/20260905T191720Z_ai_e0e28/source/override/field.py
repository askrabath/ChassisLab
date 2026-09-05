"""Appendix A reconstruction, not an official CAD import. All lengths SI."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from ..physics import element

VERSION='override-v2-reconstruction-1'
PIN_H=.165; PIN_TIP_R=.0178; PIN_BASE_R=.0298; PIN_FLANGE_R=.04015
CUP_H=.1645; CUP_OUTER_R=.0401; CUP_NECK_R=.0295; WALL=.002
GOAL_R=.03005; GOAL_H=.0825
FIELD_SIZE=3.5664
# Appendix A10, lower-left field origin. Height in meters.
GOALS=[(.5871,1.1851,.0825,'red'),(1.1851,.5871,.0825,'red'),
       (2.3813,2.9793,.0825,'blue'),(2.9793,2.3813,.0825,'blue'),
       (.5871,2.3813,.1465,'neutral'),(1.1851,2.9793,.1465,'neutral'),
       (2.3813,.5871,.1465,'neutral'),(2.9793,1.1851,.1465,'neutral'),
       (1.7832,1.7832,.2227,'neutral')]
COLORS={'red':'.85 .1 .15 1','blue':'.1 .4 .85 1','yellow':'.95 .8 .12 1','neutral':'.2 .23 .27 1'}


def mesh(asset,name,vertices):
    element(asset,'mesh',name=name,vertex=' '.join(str(x) for v in vertices for x in v))


def frustum(asset,body,name,z0,z1,r0,r1,mass,color):
    vertices=[[r*math.cos(a),r*math.sin(a),z] for z,r in [(z0,r0),(z1,r1)] for a in np.linspace(0,2*math.pi,24,endpoint=False)]
    mesh(asset,name,vertices)
    element(body,'geom',name=name,type='mesh',mesh=name,mass=mass,rgba=color)


def hollow(asset,body,name,z0,z1,outer0,outer1,inner0,inner1,mass,color,n=24):
    """Convex wall sectors, never a single convex hull across the cavity."""
    for i in range(n):
        a=i*2*math.pi/n;b=(i+1)*2*math.pi/n
        vertices=[[r*math.cos(t),r*math.sin(t),z] for z,ro,ri in [(z0,outer0,inner0),(z1,outer1,inner1)] for r in [ro,ri] for t in [a,b]]
        mesh(asset,f'{name}_{i}',vertices)
        element(body,'geom',name=f'{name}_{i}',type='mesh',mesh=f'{name}_{i}',mass=mass/n,rgba=color)


def pin(asset,world,name,pos,colors=('red','blue'),quat=None):
    attrs={'quat':quat} if quat else {}
    b=element(world,'body',name=name,pos=' '.join(map(str,pos)),**attrs);element(b,'freejoint',name=name+'_free')
    for sign,color in [(1,colors[0]),(-1,colors[1])]:
        z=sorted([sign*.008,sign*.066313])
        radii=(PIN_BASE_R,PIN_TIP_R) if sign==1 else (PIN_TIP_R,PIN_BASE_R)
        frustum(asset,b,f'{name}_taper_{sign}',*z,*radii,.038,COLORS[color])
        element(b,'geom',name=f'{name}_tip_{sign}',type='cylinder',pos=f'0 0 {sign*(.066313+.0825)/2}',size=f'{PIN_TIP_R} {(.0825-.066313)/2}',mass=.008,rgba=COLORS[color])
        element(b,'geom',name=f'{name}_flange_{sign}',type='cylinder',pos=f'0 0 {sign*.004}',size=f'{PIN_FLANGE_R} .004',mass=.012,rgba=COLORS[color])
    return b


def cup(asset,world,name,pos):
    b=element(world,'body',name=name,pos=' '.join(map(str,pos)));element(b,'freejoint',name=name+'_free')
    hollow(asset,b,name+'_lower',-CUP_H/2,0,CUP_OUTER_R,CUP_NECK_R,CUP_OUTER_R-WALL,CUP_NECK_R-WALL,.05,'.5 .7 .8 .45')
    hollow(asset,b,name+'_upper',0,CUP_H/2,CUP_NECK_R,CUP_OUTER_R,CUP_NECK_R-WALL,CUP_OUTER_R-WALL,.05,'.18 .2 .23 1')
    # Unspecified internal divider thickness is an explicit reconstruction assumption.
    element(b,'geom',name=name+'_divider',type='cylinder',size=f'{CUP_NECK_R} .001',mass=.008,rgba='.25 .25 .25 1')
    return b


def goal(asset,world,name,pos,height=GOAL_H,color='red'):
    b=element(world,'body',name=name,pos=' '.join(map(str,pos)))
    pedestal=height-GOAL_H
    if pedestal>0:
        element(b,'geom',name=name+'_pedestal',type='cylinder',pos=f'0 0 {pedestal/2}',size=f'.0819 {pedestal/2}',rgba=COLORS[color])
    element(b,'geom',name=name+'_bottom',type='cylinder',pos=f'0 0 {pedestal+.005}',size='.0819 .005',rgba=COLORS[color])
    hollow(asset,b,name+'_wall',pedestal+.010,height,.07125,.0444,GOAL_R,GOAL_R,.3,COLORS[color])
    return b


def environment(root,full=False,friction=.65):
    asset=root.find('asset')
    if asset is None:asset=element(root,'asset')
    world=root.find('worldbody')
    if world is None:world=element(root,'worldbody')
    for g in list(world.findall('geom')):
        if g.get('name')=='floor':world.remove(g)
    element(world,'geom',name='floor',type='plane',size='4 4 .1',rgba='.26 .3 .34 1',friction=f'{friction} .002 .0001')
    if full:
        for i,(x,y,h,c) in enumerate(GOALS):goal(asset,world,f'goal_{i}',(x,y,0),h,c)
        for i,(x,y,sx,sy) in enumerate([(FIELD_SIZE/2,-.0254,FIELD_SIZE/2,.0254),(FIELD_SIZE/2,FIELD_SIZE+.0254,FIELD_SIZE/2,.0254),(-.0254,FIELD_SIZE/2,.0254,FIELD_SIZE/2),(FIELD_SIZE+.0254,FIELD_SIZE/2,.0254,FIELD_SIZE/2)]):
            element(world,'geom',name=f'wall_{i}',type='box',pos=f'{x} {y} .1465',size=f'{sx} {sy} .1465',rgba='.6 .6 .63 1')
        for i,(x,y,angle) in enumerate([(FIELD_SIZE/2,0,0),(FIELD_SIZE/2,FIELD_SIZE,0),(0,FIELD_SIZE/2,math.pi/2),(FIELD_SIZE,FIELD_SIZE/2,math.pi/2)]):
            b=element(world,'body',name=f'toggle_{i}',pos=f'{x} {y} .31',euler=f'0 0 {angle}')
            element(b,'joint',name=f'toggle_hinge_{i}',type='hinge',axis='1 0 0',limited='true',range='-2.0944 2.0944',damping=.1,frictionloss=.025)
            for k,c in enumerate(['yellow','red','blue']):
                a=k*2*math.pi/3
                element(b,'geom',name=f'toggle_{i}_{k}',type='box',pos=f'0 {.017*math.sin(a)} {.017*math.cos(a)}',euler=f'{-a} 0 0',size='.3301 .02 .006',mass=.12,rgba=COLORS[c])
        # Inspection layout of representative objects, not a complete match reset.
        for i,(x,y) in enumerate([(1.1851,1.1851),(2.3813,2.3813),(.5871,.5871)]):
            pin(asset,world,f'inspection_pin_{i}',(x,y,PIN_H/2+.002))
            cup(asset,world,f'inspection_cup_{i}',(x+.15,y,CUP_H/2+.002))
    return asset,world


def nesting(pin_pos,pin_axis,container_xy,rim,bottom,radius=GOAL_R):
    """Conservative supported upright subset of SC2, not general match scoring.

    Both half-end centers inside the opening means invalid two-half occupancy.
    Mere distance to a goal cannot create a positive result.
    """
    p=np.asarray(pin_pos);a=np.asarray(pin_axis)
    ends=[p+a*s*PIN_H/2 for s in (-1,1)]
    contained=[bool(np.linalg.norm(e[:2]-container_xy)+PIN_TIP_R < radius and bottom-.002<e[2]<rim-.001) for e in ends]
    return bool(sum(contained)==1 and abs(a[2])>.90)
