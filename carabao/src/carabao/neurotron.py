#===============================================================================
# carabao.neurotron package: copyright: Neuronycs 2023
#===============================================================================
"""
Module carabao.neurotron supports the following classes:
    class Pulse
    class Terminal

Example 1:
    Test a Pulse(1,2) module (1 lag cycle, 2 duty cycles) with a delta
    sequence u = [1,0,0,0,0]. The expected response is: y = [0,1,1,0,0]

pls = Pulse(1,2,name="Pulse(1,2):")
y0 = pls.feed(u0:=1)
y1 = pls.feed(u1:=0)
y2 = pls.feed(u2:=0)
y3 = pls.feed(u3:=0)
y4 = pls.feed(u4:=0)
((u0,u1,u2,u3,u4),(y0,y1,y2,y3,y4))
Pulse(1,2): 1 -> ([1,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 0 -> ([0,0],1/2) -> 1
Pulse(1,2): 0 -> ([0,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,0],0/2) -> 0
((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))

Example 2:
    Similar case but retrigger the pulse module before the niveau collapses.
    Input sequence: u = [1,0,0,0,0], desired response: y = [0,1,1,1,1]

pls = Pulse(1,2,log="Pulse(1,2):")
y0 = pls.feed(u0:=1)
y1 = pls.feed(u1:=0)
y2 = pls.feed(u2:=1)
y3 = pls.feed(u3:=0)
y4 = pls.feed(u4:=0)
((u0,u1,u2,u3,u4),(y0,y1,y2,y3,y4))
Pulse(1,2): 1 -> ([1,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 1 -> ([1,0],1/2) -> 1
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 0 -> ([0,0],1/2) -> 1
((1, 0, 1, 0, 0), (0, 1, 1, 1, 1))

Example 3:
    Create a Terminal object and demonstrate its functionality.

>>> par,token = toy('sarah')
>>> excite = Terminal(par[0].w[0],par[0].theta,'excite')
>>> print(excite)
Terminal('excite',#[1 1 0 1 1 1 0 1 0 1],6)
"""

#===============================================================================
# imports
#===============================================================================

from numpy import array
from ypstruct import struct
from carabao.util import repr

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse:
    """
    pulse: pulse unit
    >>> u=Pulse(2,3)
    >>> for i in range(6): o = u(int(i<1),'u%g:'%i)
    u0:  1 -> ([1,0,0],0/3) -> 0
    u1:  0 -> ([0,1,0],0/3) -> 0
    u2:  0 -> ([0,0,1],3/3) -> 1
    u3:  0 -> ([0,0,0],2/3) -> 1
    u4:  0 -> ([0,0,0],1/3) -> 1
    u5:  0 -> ([0,0,0],0/3) -> 0
    >>> i = u.inp()                     # retrieve recent input
    >>> o = u.out()                     # get pulse output
    >>> u.set(1)                        # set output 1 (over full duty)
    """
    def __init__(self,lag,duty,name=None):
        def zeros(n): return [0 for k in range(0,n)]
        self.name = name                # name header
        self.n = duty                  # duty = pulse length
        self.s = zeros(lag+1)          # shift register
        self.c = 0                     # counter

    def feed(self,u):
        self.s = [u] + self.s[:-1]
        self.c = self.n if self.s[-1] > 0 else max(0,self.c-1)
        if self.name is not None: print(self)
        return self.out()

    def inp(self): return self.s[0]
    def out(self): return (self.c > 0) + 0
    def set(self,val,log=None):
        self.c = self.n if val > 0 else 0
        if log is not None:
            print(log,self)

    def __call__(self,u,log=None):
        y = self.feed(u)
        if log is not None:
            print(log,self)
        return y

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self;  body = "(%s,%g/%g)" % (string(o.s),o.c,o.n)
        name = o.name if o.name is not None else ""
        return name + " %g -> " % o.inp() + body +  " -> %g" % o.out()

#===================================================================================
# class: Synapses
#===================================================================================

class Synapses:
    """
    class Synapses: terminal selector
    >>> K = [[10,11,12],[10,11,12]];  P = [[.5,.4,.1],[.6,.2,.3]];  eta = 0.5
    >>> syn = Synapses(K,P,eta,log='Synapses:')
    >>> print(syn)
    {#[10 11 12; 10 11 12], #[0.5 0.4 0.1; 0.6 0.2 0.3] @ 0.5}
    >>> V = syn(x:=[0,0,0,0,0,0,0,0,0,0,1,1,0])
    Synapses: [0 0 0 0 0 0 0 0 0 0 1 1 0] -> #[1 0 0; 1 0 0] -> #[1 0 0; 1 0 0]
    """
    def __init__(self,K,P,eta=0.5,log=None):
        def matrix(X):
            X = array(X)
            return X if len(X.shape) > 1 else array([X])

        self.K = matrix(K)             # index matrix for selector
        self.P = matrix(P)             # permanence matrix
        self.eta = eta                 # synaptic threshold
        self.log = log                 # log header (no logging if log=None)
        #if log is not None: print(self)

    def weight(self):
        W = (self.P >= self.eta)*1;
        return W

    def __call__(self,x,log=None):     # feed network state to synapses
        eta = self.eta;  K = self.K;  V = 0*K;
        W = self.weight()
        for i in range(0,K.shape[0]):
            for j in range(0,K.shape[1]):
                V[i,j] = x[K[i,j]] if W[i,j] > 0 else 0
        if self.log is not None:
            print(self.log,repr(x),"->",repr(W),"->",repr(V))
        return V

    def __repr__(self):
        head = "%s " % self.log if self.log is not None else ""
        return "{" + repr(self.K) + ", " + repr(self.P) + " @ %g}" % self.eta

#===============================================================================
# class: Terminal
#===============================================================================

class Terminal:
    """
    class Terminal: to model a McCulloch-Pitts-type synapse terminal
    >>> w = [1,1,0,1,1,1,0,1,0,1]   # binary weights
    >>> theta = 6                   # spiking threshold
    >>> excite = Terminal(w,theta,'excite')
    >>> print(excite)
    Terminal('excite',#[1 1 0 1 1 1 0 1 0 1],6)
    """

    def __init__(self,W,theta,name=None):
        def matrix(X):
            X = array(X)
            return X if len(X.shape) > 1 else array([X])

        self.synapses = None        # synapses object
        self.W = matrix(W)
        self.theta = theta          # spiking threshold
        self.name = name              # name string
        #if name is not None:
        #    print(self)

    def empower(self,V,log=None):   # determine empowerment
        if self.synapses is not None:
            self.W = self.synapses.weight()
        E = self.W * V
        if log is not None:
            print(log,repr(V),"->",repr(E))
        return E

    def spike(self,E,log=None):     # spike function
        S = array([sum(E[k]) for k in range(0,E.shape[0])])
        s = (S >= self.theta)*1
        if log is not None:
            print(log,repr(E),"->",repr(s))
        return s

    def __call__(self,x,log=None):      # feed x vector to terminal
        if self.synapses is None:
            return ([],[])
        E = self.empower(x)
        s = self.spike(E)
        if log is not None:
            if self.synapses is None:
                print(log,repr(x),"->",repr(E),"->",repr(s))
            else:
                W = self.synapses.weight()
                print(log,repr(x),"->",repr(W),"->",repr(E),"->",repr(s))
        if len(s) == 1:
           return s.item()  # (s,E)
        return s.any()*1

    def __repr__(self):
        head = "(" if self.name is None else "('%s'," % self.name
        par = head + repr(self.W) + "," + "%g)"%self.theta;
        syn = "" if self.synapses is None \
                 else " @ " + self.synapses.__repr__()
        return "Terminal" + par + syn

#===============================================================================
# helper: set up toy stuff
#===============================================================================

def toy(mode):
    """
    toy(): setup toy stuff
    >>> par,token = toy('sarah') # get params for 'sarah' app
    >>> excite,depress,predict = par
    """
    def bundle(obj,n):                      # create a bunch of object as a list
        return [obj for k in range(0,n)]
    idx = [k for k in range(0,13)]
    prm = [.3,.4,.1, .5,.2,.3, .1,.7,.3,]

    if mode == 'sarah':
        token = {'Sarah':[1,1,0,1,1,1,0,1,0,1],
                 'loves':[0,1,1,1,0,1,1,0,1,1],
                 'music':[1,1,1,0,0,1,0,1,1,1]}

        f1 = token['Sarah']
        f2 = token['loves']
        f3 = token['music']

        e = struct();                       # excitation terminal parameters
        e.w = [f1,f2,f3]                    # excitation weights
        e.k = bundle(idx[:10],3)            # selects feedforward part of x
        e.p = [f1,f2,f3];
        e.theta = 6                         # spiking threshold
        e.eta = 0.5                         # synaptic threshold

        d = struct()                        # depression terminal parameters
        d.w = bundle([1,1,0],3)             # depression weights
        d.g = bundle([10,11,12],3);         # group indices
        d.p = bundle([1,1,1],3);            # all depression permanences are 1
        d.theta = 1                         # depression threshold
        d.eta = 0.5                         # synaptic threshold

        p = struct()                        # prediction terminal parameters
        p.W = bundle([[1,0,0],[0,1,1]],3)   # prediction weights
        p.K = bundle([idx[10:],idx[10:]],3)
        p.P = [[prm[0:3],prm[0:3]],
               [prm[3:6],prm[0:3]],
               [prm[6:9],prm[0:3]]]
        p.theta = 1                         # prediction threshold
        p.eta = 0.5                         # synaptic threshold

        return (e,d,p),token

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
