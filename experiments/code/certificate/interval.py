"""Small outward-rounded Decimal interval kernel, Python standard library only.

All certificate inputs are decimal strings or integers. No floating-point input
is accepted. Basic operations use directed rounding. Decimal exp/sqrt are
correctly rounded to nearest; one representable number is added on either side.
"""
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN

PREC = 42
DN = Context(prec=PREC, rounding=ROUND_FLOOR)
UP = Context(prec=PREC, rounding=ROUND_CEILING)
NEAR = Context(prec=PREC, rounding=ROUND_HALF_EVEN)

def D(x):
    if isinstance(x, Decimal): return x
    assert isinstance(x, (str,int)), 'Use exact decimal strings, not floats'
    return Decimal(x)

class I:
    __slots__ = ('lo','hi')
    def __init__(self,lo,hi=None):
        if isinstance(lo,I):self.lo,self.hi=lo.lo,lo.hi;return
        self.lo=D(lo);self.hi=D(lo if hi is None else hi)
        assert self.lo<=self.hi
    def __add__(self,y):
        y=I(y);return I(DN.add(self.lo,y.lo),UP.add(self.hi,y.hi))
    __radd__=__add__
    def __neg__(self):return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,y):return self+-I(y)
    def __rsub__(self,y):return I(y)+-self
    def __mul__(self,y):
        y=I(y);pairs=[(a,b) for a in (self.lo,self.hi) for b in (y.lo,y.hi)]
        return I(min(DN.multiply(a,b) for a,b in pairs),max(UP.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def __truediv__(self,y):
        y=I(y);assert y.lo>0 or y.hi<0,'division interval contains zero'
        return self*I(DN.divide(D(1),y.hi),UP.divide(D(1),y.lo))
    def __rtruediv__(self,y):return I(y)/self
    def sq(self):
        if self.lo>=0:return I(DN.multiply(self.lo,self.lo),UP.multiply(self.hi,self.hi))
        if self.hi<=0:return (-self).sq()
        return I(0,max(UP.multiply(self.lo,self.lo),UP.multiply(self.hi,self.hi)))
    def sqrt(self):
        assert self.lo>=0
        low=NEAR.sqrt(self.lo);high=NEAR.sqrt(self.hi)
        return I(max(D(0),DN.next_minus(low)),UP.next_plus(high))
    def exp(self):
        return I(max(D(0),DN.next_minus(NEAR.exp(self.lo))),UP.next_plus(NEAR.exp(self.hi)))
    def absmax(self):return max(self.lo.copy_abs(),self.hi.copy_abs())
    def __repr__(self):return f'[{self.lo}, {self.hi}]'

def norm(v):return sum((x.sq() for x in v),I(0)).sqrt()
def dot(a,b):return sum((x*y for x,y in zip(a,b)),I(0))
def box(c,r):return [I(x)-I(D(r).copy_negate(),D(r)) for x in c]
def eigbounds(H):
    # Extremal eigenvalues are monotone in each diagonal entry, and are
    # minimized/maximized by the largest absolute off-diagonal entry.
    b=I(H[0][1].absmax())
    a,d=I(H[0][0].lo),I(H[1][1].lo)
    low=((a+d-((a-d).sq()+4*b.sq()).sqrt())/2).lo
    a,d=I(H[0][0].hi),I(H[1][1].hi)
    high=((a+d+((a-d).sq()+4*b.sq()).sqrt())/2).hi
    return low,high

M_STR=[['1.38','-1.03'],['-0.22','-0.19'],['-1.16','1.22']]
A_STR=[['-0.08','0.24'],['-0.95','-0.31']]

class Model:
    def __init__(self,delta='0.0001',noise='0.001'):
        d=I(D(delta).copy_negate(),D(delta))
        self.M=[[I(t)+d for t in row] for row in M_STR]
        self.A=[[I(t)+d for t in row] for row in A_STR]
        self.tau=I('0.15')+d;self.lam=I(1)+d
        self.pi=[I(1)+d for _ in range(3)] # unnormalized positive masses
        self.gamma=I('0.35')+d;self.noise=I(D(noise).copy_negate(),D(noise))
        self.B=[[sum((self.A[k][i]*self.A[k][j] for k in range(2)),I(0)) for j in range(2)] for i in range(2)]
    def y(self,j):return [dot(row,self.M[j])+self.noise for row in self.A]
    def init(self,j):
        y=self.y(j)
        return [sum((self.A[k][i]*y[k] for k in range(2)),I(0)) for i in range(2)]
    def quantities(self,x,s):
        v=self.tau.sq()+s.sq();q=s.sq()/v
        weights=[]
        for i in range(3):
            denom=I(1)
            for j in range(3):
                if i==j:continue
                # Difference of quadratic logits, linear in x.
                dm=[self.M[j][k]-self.M[i][k] for k in range(2)]
                mid=[(self.M[j][k]+self.M[i][k])/2 for k in range(2)]
                z=dot([x[k]-mid[k] for k in range(2)],dm)/v
                denom+=self.pi[j]/self.pi[i]*z.exp()
            weights.append(1/denom)
        mean=[sum((weights[i]*self.M[i][k] for i in range(3)),I(0)) for k in range(2)]
        return v,q,weights,mean
    def g(self,x,j,s):
        v,q,w,mean=self.quantities(x,s)
        ax_y=[dot(self.A[k],[x[t]-self.M[j][t] for t in range(2)])-self.noise for k in range(2)]
        return [sum((self.A[k][i]*ax_y[k] for k in range(2)),I(0))+self.lam*q*(x[i]-mean[i]) for i in range(2)]
    def h(self,x,s):
        v,q,w,mean=self.quantities(x,s)
        C=[[I(0) for _ in range(2)] for _ in range(2)]
        for i in range(3):
            for j in range(i):
                dm=[self.M[i][k]-self.M[j][k] for k in range(2)]
                for a in range(2):
                    for b in range(a,2):
                        C[a][b]+=w[i]*w[j]*(dm[a].sq() if a==b else dm[a]*dm[b])
                        C[b][a]=C[a][b]
        return [[self.B[a][b]+self.lam*q*(I(int(a==b))-C[a][b]/v) for b in range(2)] for a in range(2)]
    def gs(self,x,s):
        v,q,w,mean=self.quantities(x,s)
        cov=[I(0),I(0)]
        for i in range(3):
            for j in range(i):
                dm=[self.M[i][k]-self.M[j][k] for k in range(2)]
                mid=[(self.M[i][k]+self.M[j][k])/2 for k in range(2)]
                dd=-2*dot([x[k]-mid[k] for k in range(2)],dm)
                for k in range(2):cov[k]+=w[i]*w[j]*dm[k]*dd
        qp=2*s*self.tau.sq()/v.sq()
        return [self.lam*(qp*(x[k]-mean[k])-q*s/v.sq()*cov[k]) for k in range(2)]
    def contraction(self,x,s):
        m,L=eigbounds(self.h(x,s))
        # H is symmetric, so the Jacobian spectral norm is bounded by endpoints.
        rho=max((1-self.gamma*I(m)).absmax(),(1-self.gamma*I(L)).absmax())
        return m,L,rho
    def step(self,x,j,s):return [x[k]-self.gamma*g for k,g in enumerate(self.g(x,j,s))]
