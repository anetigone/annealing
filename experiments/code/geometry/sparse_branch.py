"""Sparse polyline tubes: scalar competition geometry and a residual derivative.

The candidate generator is floating point; the cell checker uses only outward
rounded intervals. It covers the entire segment, not a sample of its points.
"""
from geometry import *

def segment_check(model,j,a,b,ca,cb,R='0.03'):
    a,b=I(a),I(b);s=I(b.lo,a.hi);h=a-b
    ca=[I(x) for x in ca];cb=[I(x) for x in cb]
    mid=scale(I('.5'),add(ca,cb));slope=scale(1/h,sub(ca,cb))
    xr=add(mid,scale(I(-D('.5'),D('.5'))*h,slope))
    tube=[x+I(D(R).copy_negate(),D(R)) for x in xr]
    m,L,competition=curvature_trace(model,tube,s)
    dref=add(matvec(model.h(xr,s),slope),model.gs(xr,s))
    gmid=model.g(mid,j,(a+b)/2)
    residual=norm(add(gmid,scale(I(-D('.5'),D('.5'))*h,dref))).hi
    rootdist=(I(residual)/I(m)).hi if m>0 else D('Infinity')
    return {'m':m,'L':L,'competition':competition,'residual':residual,'rootdist':rootdist,
       'pass':m>0 and rootdist<D(R)}

