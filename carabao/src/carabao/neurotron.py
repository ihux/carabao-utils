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

pls = Pulse(1,2,log="Pulse(1,2):")
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

#===============================================================================
# class: Terminal
#===============================================================================

class Terminal:
    """
    class Terminal: to model a McCulloch-Pitts-type synapse terminal
    >>> w = [1,1,0,1,1,1,0,1,0,1]   # binary weights
    >>> theta = 6                   # spiking threshold
    >>> excite = Terminal(w,theta,None,'excite')
    Terminal('excite',#[1 1 0 1 1 1 0 1 0 1],6)
    """

    def __init__(self,W,theta,select=None,log=None):
        def matrix(X):
            X = array(X)
            return X if len(X.shape) > 1 else array([X])

        self.select = select        # select unit
        self.W = matrix(W)
        self.theta = theta          # spiking threshold
        self.log = log              # log string
        if log is not None:
            print(self)

    def empower(self,V):            # determine empowerment
        return self.W * array(V)

    def spike(self,E):              # spike function
        S = array([sum(E[k]) for k in range(0,E.shape[0])])
        return (S >= self.theta)*1

    def __repr__(self):
        head = "(" if self.log is None else "('%s'," % self.log
        par = head + repr(self.W) + "," + "%g)"%self.theta;
        return "Terminal" + par

#===============================================================================
# helper: set up toy stuff
#===============================================================================

def toy(mode):
    """
    toy(): setup toy stuff
    >>> excite,predict,depress,token = toy('sarah') # get params for 'sarah' app
    """
    def bundle(obj,n):                      # create a bunch of object as a list
        return [obj for k in range(0,n)]

    if mode == 'sarah':
        token = {'Sarah':[1,1,0,1,1,1,0,1,0,1],
                 'likes':[0,1,1,1,0,1,1,0,1,1],
                 'music':[1,1,1,0,0,1,0,1,1,1]}

        f1 = token['Sarah']
        f2 = token['likes']
        f3 = token['music']

        e = struct();                       # excitation terminal parameters
        e.w = [f1,f2,f3]                    # excitation weights
        e.theta = 6                         # spiking threshold

        p = struct()                        # prediction terminal parameters
        p.W = bundle([[1,0,0],[0,1,1]],3)   # prediction weights
        p.theta = 2                         # prediction threshold

        d = struct()                        # depression terminal parameters
        d.w = bundle([1,1,0],1)             # depression weights
        d.theta = 1                         # depression threshold

        return (e,d,p,token)

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
