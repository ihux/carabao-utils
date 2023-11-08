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

>>> par = toy('sarah')
>>> excite = Terminal(par[0].w[0],par[0].theta,None,'excite')
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
    >>> o = Pulse(lag=2,duty=3)             # mandatory args
    >>> o = Pulse(lag=2,duty=3,log="P2/3")  # set log text
    >>> y = o.feed(u:=1)                    # feed new input, receive new output
    P2/3 1 -> ([1,0,0],0/3) -> 0
    >>> u = o.inp()                         # retrieve recent input
    >>> y = o.out()                         # get pulse output
    """
    def __init__(self,lag,duty,log=None):
        def zeros(n): return [0 for k in range(0,n)]
        self.head = log                # log header
        self.n = duty                  # duty = pulse length
        self.s = zeros(lag+1)          # shift register
        self.c = 0                     # counter

    def feed(self,u):
        self.s = [u] + self.s[:-1]
        self.c = self.n if self.s[-1] > 0 else max(0,self.c-1)
        if self.head is not None: print(self)
        return self.out()

    def inp(self): return self.s[0]
    def out(self): return (self.c > 0) + 0

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self;  body = "(%s,%g/%g)" % (string(o.s),o.c,o.n)
        return o.head + " %g -> " % o.inp() + body +  " -> %g" % o.out()

#===================================================================================
# class: Synapses
#===================================================================================

class Synapses:
    """
    class Synapses: terminal selector
    >>> K = [[10,11,12],[10,11,12]];  P = [[.5,.4,.1],[.6,.2,.3]];  eta = 0.5
    >>> syn = Synapses(K,P,eta,log='Synapses:')
    {#[10 11 12; 10 11 12], #[0.5 0.4 0.1; 0.6 0.2 0.3] @ 0.5}
    >>> V = syn.feed(x:=[0,0,0,0,0,0,0,0,0,0,1,1,0])
    Synapses: [0 0 0 0 0 0 0 0 0 0 1 1 0]  ->  #[1 0 0; 1 0 0]
    """
    def __init__(self,K,P,eta=0.5,log=None):
        def matrix(X):
            X = array(X)
            return X if len(X.shape) > 1 else array([X])

        self.K = matrix(K)             # index matrix for selector
        self.P = matrix(P)             # permanence matrix
        self.eta = eta                 # synaptic threshold
        self.log = log                 # log header (no logging if log=None)
        if log is not None: print(self)

    def feed(self,x):                  # feed network state to synapses
        eta = self.eta;  K = self.K;  V = 0*K;
        for i in range(0,K.shape[0]):
            for j in range(0,K.shape[1]):
                V[i,j] = x[K[i,j]] if self.P[i,j] >= eta else 0
        if self.log is not None:
            print(self.log,repr(x)," -> ",repr(V))
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
    >>> excite = Terminal(w,theta,None,'excite')
    >>> print(excite)
    Terminal('excite',#[1 1 0 1 1 1 0 1 0 1],6)
    """

    def __init__(self,W,theta,synapses=None,name=None):
        def matrix(X):
            X = array(X)
            return X if len(X.shape) > 1 else array([X])

        self.synapses = synapses    # synapses instance
        self.W = matrix(W)
        self.theta = theta          # spiking threshold
        self.name = name              # name string
        #if name is not None:
        #    print(self)

    def empower(self,V):            # determine empowerment
        return self.W * array(V)

    def spike(self,E):              # spike function
        S = array([sum(E[k]) for k in range(0,E.shape[0])])
        return (S >= self.theta)*1

    def feed(self,x):               # feed x vector to terminal
        if self.synapses is not None:
            return ([],[])
        E = self.empower(x)
        s = self.spike(E)
        return (s,E)

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
    >>> excite,depress,predict,token = toy('sarah') # get params for 'sarah' app
    """
    def bundle(obj,n):                      # create a bunch of object as a list
        return [obj for k in range(0,n)]
    idx = [k for k in range(0,13)]
    prm = [.5,.4,.1,.6,.2,.3]

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
        p.P = bundle([prm[0:3],prm[3:6]],3)
        p.theta = 2                         # prediction threshold
        p.eta = 0.5                         # synaptic threshold

        return (e,d,p,token)

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
