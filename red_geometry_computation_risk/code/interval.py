"""Small outward-rounded Decimal interval kernel (not formally verified)."""
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN

PRECISION = 50
DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR)
UP = Context(prec=PRECISION, rounding=ROUND_CEILING)
NEAR = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)

def dec(x):
    if isinstance(x, float):
        raise TypeError('Binary floats are not certificate inputs')
    return x if isinstance(x, Decimal) else Decimal(x)

class I:
    __slots__ = ('lo', 'hi')
    def __init__(self, lo=0, hi=None):
        if isinstance(lo, I):
            self.lo, self.hi = lo.lo, lo.hi
        else:
            self.lo, self.hi = dec(lo), dec(lo if hi is None else hi)
        if self.lo > self.hi:
            raise ValueError('Reversed interval')
    def __add__(self, b):
        b = I(b)
        return I(DOWN.add(self.lo,b.lo), UP.add(self.hi,b.hi))
    __radd__ = __add__
    def __neg__(self):
        return I(self.hi.copy_negate(), self.lo.copy_negate())
    def __sub__(self,b): return self + (-I(b))
    def __rsub__(self,b): return I(b) + (-self)
    def __mul__(self,b):
        b=I(b)
        pairs=[(a,c) for a in (self.lo,self.hi) for c in (b.lo,b.hi)]
        return I(min(DOWN.multiply(a,c) for a,c in pairs),
                 max(UP.multiply(a,c) for a,c in pairs))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=I(b)
        if b.lo <= 0 <= b.hi: raise ZeroDivisionError('Interval contains zero')
        return self * I(DOWN.divide(Decimal(1),b.hi),UP.divide(Decimal(1),b.lo))
    def __rtruediv__(self,b): return I(b)/self
    def sq(self):
        high=max(UP.multiply(self.lo,self.lo),UP.multiply(self.hi,self.hi))
        low=Decimal(0) if self.lo <= 0 <= self.hi else min(DOWN.multiply(self.lo,self.lo),DOWN.multiply(self.hi,self.hi))
        return I(low,high)
    def exp(self):
        return I(NEAR.next_minus(NEAR.exp(self.lo)),NEAR.next_plus(NEAR.exp(self.hi)))
    def sqrt(self):
        if self.lo < 0: raise ValueError('Negative sqrt interval')
        low=Decimal(0) if self.lo==0 else NEAR.next_minus(NEAR.sqrt(self.lo))
        return I(low,NEAR.next_plus(NEAR.sqrt(self.hi)))
    def absmax(self): return max(self.lo.copy_abs(),self.hi.copy_abs())
    def json(self): return [str(self.lo),str(self.hi)]
    def __repr__(self): return f'I({self.lo},{self.hi})'

def dot(a,b): return sum((x*y for x,y in zip(a,b)),I(0))
def norm(a): return sum((x.sq() for x in a),I(0)).sqrt()
def matvec(A,x): return [dot(row,x) for row in A]
def transpose(A): return [list(x) for x in zip(*A)]
def matmul(A,B): return [[dot(a,b) for b in transpose(B)] for a in A]
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def eigbounds(H):
    # H[0][1] encloses the same exact symmetric entry as H[1][0].
    tr=H[0][0]+H[1][1]
    disc=((H[0][0]-H[1][1]).sq()+4*H[0][1].sq()).sqrt()
    return ((tr-disc)/2).lo,((tr+disc)/2).hi
