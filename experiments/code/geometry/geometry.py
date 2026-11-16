"""Interpretable, outward-rounded geometric bounds for the RED/GMM objective."""
from interval_core import I,D,UP,DN,Model as BaseModel,norm,dot,box,eigbounds

def q_range(tau,s):
    """q increases with positive sigma and decreases with positive tau."""
    assert tau.lo>0 and s.lo>0
    lo=(I(s.lo).sq()/(I(tau.hi).sq()+I(s.lo).sq())).lo
    hi=(I(s.hi).sq()/(I(tau.lo).sq()+I(s.hi).sq())).hi
    return I(lo,hi)

class Model(BaseModel):
    def quantities(self,x,s):
        v,q,w,mean=super().quantities(x,s)
        return v,q_range(self.tau,s),w,mean

def matvec(A,x):return [dot(r,x) for r in A]
def matmul(A,B):return [[sum((A[i][k]*B[k][j] for k in range(len(B))),I(0)) for j in range(len(B[0]))] for i in range(len(A))]
def inv2(A):
    det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    assert det.lo>0
    return [[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def add(a,b):return [x+y for x,y in zip(a,b)]
def scale(s,v):return [s*x for x in v]

def matrices(model,s):
    v=model.tau.sq()+s.sq();c=model.lam*q_range(model.tau,s)
    Q=[[model.B[i][j]+(c if i==j else I(0)) for j in range(2)] for i in range(2)]
    return v,c,Q,inv2(Q)

def lengths(model,s):
    v,c,Q,Qi=matrices(model,s)
    return {(i,j):c/v*dot(sub(model.M[i],model.M[j]),matvec(Qi,sub(model.M[i],model.M[j]))) for i in range(3) for j in range(i)}

def curvature_trace(model,x,s):
    v,c,w,mean=model.quantities(x,s)
    ell=lengths(model,s)
    upper=sum((I(0,min((w[i]*w[j]).hi,D('.25')))*value for (i,j),value in ell.items()),I(0)).hi
    _,_,Q,Qi=matrices(model,s);alpha,L=eigbounds(Q)
    return (I(alpha if upper<=1 else L)*(1-I(upper))).lo,L,upper

def single_center(model,target,component,s):
    """z_l = mu_l + Q^-1 B(mu_t-mu_l) + Q^-1 A^T eta."""
    v,c,Q,Qi=matrices(model,s)
    At=[[model.A[j][i] for j in range(2)] for i in range(2)]
    W=matmul(Qi,At)
    z0=add(model.M[component],matvec(matmul(Qi,model.B),sub(model.M[target],model.M[component])))
    noise=matvec(W,[model.noise,model.noise])
    return add(z0,noise)

def dominant_ball(model,target,component,s,R):
    """Closed-form center, responsibility gaps, invariance, and contraction.

    R is a Euclidean radius. The ball center varies with the fixed observation
    and model, and is not a numerical stationary point.
    """
    R=I(R);v,c,Q,Qi=matrices(model,s);alpha,L=eigbounds(Q)
    z=single_center(model,target,component,s)
    others=[j for j in range(3) if j!=component]
    ds={j:sub(model.M[j],model.M[component]) for j in others}
    ratios={};gaps={}
    for j in others:
        d=ds[j]
        # (|x-mu_j|^2-|x-mu_component|^2)/2, at the moving center.
        gap=dot(d,d)/2-dot(d,sub(z,model.M[component]))-norm(d)*R
        gaps[j]=gap
        ratios[j]=(model.pi[j]/model.pi[component]*(-gap/v).exp()).hi
    force=sum((I(r)/(1+I(r))*norm(ds[j]) for j,r in ratios.items()),I(0))
    invariance=(c*force/(I(alpha)*R)).hi
    ell=lengths(model,s);competition=I(0)
    for j,r in ratios.items():
        # Sup r/(1+r)^2 on [0,r_hi].
        prod=I('.25') if r>=D(1) else I(r)/(1+I(r)).sq()
        competition+=prod*ell[tuple(sorted((component,j),reverse=True))]
    j,k=others
    competition+=I(ratios[j])*I(ratios[k])*ell[tuple(sorted((j,k),reverse=True))]
    m=(I(alpha if competition.hi<=1 else L)*(1-competition)).lo
    dist=norm(sub(z,model.M[target]))
    return {'center':z,'gap':gaps,'ratio':ratios,'invariance':invariance,'competition':competition.hi,'m':m,'L':L,
       'error_lower':(dist-R).lo,'error_upper':(dist+R).hi}

def init_distance_to_center(model,target,component,s):
    """Preserve the common observation noise in x0-z rather than subtract boxes."""
    _,c,Q,Qi=matrices(model,s)
    ImQi=[[I(int(i==j))-Qi[i][j] for j in range(2)] for i in range(2)]
    At=[[model.A[j][i] for j in range(2)] for i in range(2)]
    nominal=sub(matvec(matmul(ImQi,model.B),model.M[target]),scale(c,matvec(Qi,model.M[component])))
    return norm(add(nominal,matvec(matmul(ImQi,At),[model.noise,model.noise]))).hi
