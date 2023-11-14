#===============================================================================
# carabao.neurotron package: copyright: Neuronycs 2023
#===============================================================================
"""
Module carabao.neurotron supports the following classes:
    class Pulse
    class Synapses
    class Terminal
    class Monitor
    function toy()

Example:
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
from carabao.screen import Screen

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse:
    """
    pulse: pulse unit with debouncer based on state machine
    >>> u=Pulse(2,3)
    >>> for i in range(6): o = u(int(i<3),'u%g:'%i)
    u0:  1 -> (1,L1,[2,3,0]) -> 0
    u1:  1 -> (2,L2,[2,3,0]) -> 0
    u2:  1 -> (2,D3,[2,3,0]) -> 1
    u3:  0 -> (1,D2,[2,3,0]) -> 1
    u4:  0 -> (0,D1,[2,3,0]) -> 1
    u5:  0 -> (0,L0,[2,3,0]) -> 0
    >>> i = u.inp()                     # retrieve recent input
    >>> o = u.out()                     # get pulse output
    >>> u.set(1)                        # set output 1 (over full duty)
    """
    def __init__(self,lag,duty,relax=0,name=None):
        self.name = name                # name header
        self.n = [lag,duty,relax]       # phase durations
        self.s = 'L'                    # state, initialized as `lag state`
        self.x = 0                      # integrator
        self.c = 0                      # counter
        self.u = 0                      # input
        self.y = 0                      # output

    def trans(self,state):              # state transition
        l,d,r = self.n                  # get parameters
        if state == 'L':                # transition to lag state
            self.x = self.c = self.u    # init integration counter
        elif state == 'D':              # transition to duty state
            self.c = d                  # set duration counter
        elif state == 'R':              # transition to relax state
            self.c = r                  # set relax counter
        self.s = state                  # actual state change

    def integrate(self,u):
        l = self.n[0]                   # lag duration
        i = self.x + 2*u - 1            # integrator output
        self.x = max(0,min(i,l))        # limit integrator state
        return self.x,i                 # return integrator state/output

    def call(self,u):
        self.u = u;
        l,d,r = self.n                  # get parameters
        x,i = self.integrate(u)         # integrate
        if self.s == 'L':               # L: lag state (debouncing)
            self.y = int(i > l and d > 0)
            self.c = x
            if self.y > 0: self.trans('D')
        elif self.s == 'D':             # D: duty state
            if self.n[2] > 0:           # no retrigger if relax
                self.c -= 1
            elif x >= l > 0:
                self.c = d
            else:
                self.c -= 1             # count down duty duration
            self.y = int(self.c > 0)
            if self.y == 0 and r > 0:
                self.trans('R')         # transition to relax state
            elif self.y == r == 0:
                self.trans('L')         # transition to lag state
        elif self.s == 'R':             # R: relax state
            self.c -= 1                 # count down relax period
            if self.c <= 0: self.trans('L')
        if self.name is not None: print(self)
        return self.out()

    def inp(self): return self.u
    def out(self): return self.y
    def set(self,val,log=None):
        if val > 0:
            self.trans('D')
        else:
            self.trans('L')
        if log is not None:
            print(log,self)

    def __call__(self,u,log=None):
        y = self.call(u)
        if log is not None:
            print(log,self)
        return y

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self
        body = "(%g,%s%g,%s)" % (self.x,self.s,self.c,string(self.n))
        name = self.name if self.name is not None else ""
        return name + " %g -> " % self.inp() + body +  " -> %g" % self.out()

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse1:
    """
    pulse: pulse unit without debouncer (implemented with shift register)
    >>> u=Pulse1(2,3)
    >>> for i in range(6): o = u(int(i<1),'u%g:'%i)
    u0:  1 -> ([1,0,0], 0/3/0) -> 0
    u1:  0 -> ([0,1,0], 0/3/0) -> 0
    u2:  0 -> ([0,0,1], 3/3/0) -> 1
    u3:  0 -> ([0,0,0], 2/3/0) -> 1
    u4:  0 -> ([0,0,0], 1/3/0) -> 1
    u5:  0 -> ([0,0,0], 0/3/0) -> 0
    >>> i = u.inp()                     # retrieve recent input
    >>> o = u.out()                     # get pulse output
    >>> u.set(1)                        # set output 1 (over full duty)
    """
    def __init__(self,lag,duty,relax=0,name=None):
        self.name = name                # name header
        self.n = duty                   # duty = pulse length
        self.s = self.zeros(lag+1)      # shift register
        self.c = 0                      # counter
        self.r = relax

    def zeros(self,n):
        return [0 for k in range(n)]

    def call(self,u):
        if self.c < 0:                  # relax mode?
            self.s = self.zeros(len(self.s))
            self.s[0] = u;  self.c += 1
        else:
            self.s = [u] + self.s[:-1]
            if self.r > 0 and self.c == 1:
                self.c = -self.r
            elif self.r > 0 and self.c > 1:
                self.c -= 1
            else:
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
        y = self.call(u)
        if log is not None:
            print(log,self)
        return y

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self;  sgn = ' ' if o.c >= 0 else ''
        body = "(%s,%s%g/%g/%g)" % (string(o.s),sgn,o.c,o.n,o.r)
        name = o.name if o.name is not None else ""
        return name + " %g -> " % o.inp() + body +  " -> %g" % o.out()

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse2:
    """
    pulse: pulse unit with debouncer (implemented with shift register)
    >>> u=Pulse2(2,3)
    >>> for i in range(6): o = u(int(i<2),'u%g:'%i)
    u0:  1 -> ([1,0,0], 0/3/0) -> 0
    u1:  1 -> ([1,1,0], 0/3/0) -> 0
    u2:  0 -> ([0,1,1], 3/3/0) -> 1
    u3:  0 -> ([0,0,1], 2/3/0) -> 1
    u4:  0 -> ([0,0,0], 1/3/0) -> 1
    u5:  0 -> ([0,0,0], 0/3/0) -> 0
    >>> i = u.inp()                     # retrieve recent input
    >>> o = u.out()                     # get pulse output
    >>> u.set(1)                        # set output 1 (over full duty)
    """
    def __init__(self,lag,duty,relax=0,name=None):
        self.name = name                # name header
        self.n = duty                   # duty = pulse length
        self.s = self.zeros(lag+1)      # shift register
        self.c = 0                      # counter
        self.r = relax

    def zeros(self,n):
        return [0 for k in range(n)]

    def call(self,u):
        o = self
        if o.c < 0:                  # relax mode?
            o.s = o.zeros(len(o.s))
            o.s[0] = u;  o.c += 1
        else:
            o.s = [u] + o.s[:-1]
            if o.r > 0 and o.c == 1:
                o.c = -o.r
            elif o.r > 0 and o.c > 1:
                o.c -= 1
            elif len(o.s) > 1:
                o.c = o.n if sum(o.s[1:]) >= len(o.s)-1 else max(0,o.c-1)
            else:
                o.c = o.n if o.s[0] > 0 else max(0,o.c-1)
        if o.name is not None: print(o)
        return o.out()

    def inp(self): return self.s[0]
    def out(self): return (self.c > 0) + 0
    def set(self,val,log=None):
        self.c = self.n if val > 0 else 0
        if log is not None:
            print(log,self)

    def __call__(self,u,log=None):
        y = self.call(u)
        if log is not None:
            print(log,self)
        return y

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self;  sgn = ' ' if o.c >= 0 else ''
        body = "(%s,%s%g/%g/%g)" % (string(o.s),sgn,o.c,o.n,o.r)
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

#===========================================================================
# class Monitor
#===========================================================================

class Monitor:
    def __init__(self,m,n,title=None):
        self.screen = Screen('Neurons',m,n)
        if title is not None: self.title(title)
    def __call__(self,cell,i,j):
        u = cell.u.out()
        q = cell.q.out()
        x = cell.x.out()
        y = cell.y.out()
        b = cell.b.out()
        d = cell.d.out()
        l = cell.l.out()
        self.screen.neurotron((i,j),u,q,x,y,b,d,l)
    def xlabel(self,x,txt,size=None):
        self.screen.text(x,-0.75,txt)
    def title(self,txt,size=10):
        scr = self.screen
        x = (scr.n-1)/2;  y = scr.m + 0.3
        self.screen.text(x,y,txt,size=size)

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

class Neurotron:
    """
    class Neurotron: full functionality
    >>> par,sizes = toy('sarah') # epar,dpar,ppar = par
    >>> cell0 = Neurotron(k:=0,par,sizes,'cell0')
    >>> print(cell0)
    Neurotron('cell0',0)
    """
    def __init__(self,k,par,sizes,name=None):
        self.k = k
        self.sizes = sizes
        self.name = name

        epar,dpar,ppar = par

        self.excite  = Terminal(epar.w[k],epar.theta,'excite')
        self.excite.synapses = Synapses(epar.k[k],epar.p[k],epar.eta)

        self.depress = Terminal(dpar.w[k],dpar.theta,'depress')
        self.depress.synapses = Synapses(dpar.g[k],dpar.p[k],dpar.eta)

        self.predict = Terminal(ppar.W[k],ppar.theta,'predict')
        self.predict.synapses = Synapses(ppar.K[k],ppar.P[k],ppar.eta)

        self.u = Pulse(0,4,3)
        self.q = Pulse(2,1,0)
        self.x = Pulse(1,7,0)
        self.y = Pulse(1,2,0)
        self.d = Pulse(0,2,0)
        self.b = Pulse(0,2,2)
        self.l = Pulse(1,1,5)

    def __call__(self,y,log=None):
        def _or(x,y): return min(x+y,1)
        def _not(x): return (1-x)
        def _log(topic,k):
            if log is not None: return "%s%g:" % (topic,k)
            return None

        k = self.k
        c,f = self.split(y,log)

        _d = self.depress(c,_log('=> depress-',k))
        _u = self.excite(f,_log('=> excite-',k))
        _s = self.predict(c,_log('=> predict-',k))

        d = self.d(_d,_log(' - d',k))   # optional
        u = self.u(_u,_log(' - u',k))
        q = self.q( u,_log(' - q',k))

        _b = _not(_d) * q
        b = self.b(_b,_log(' - b',k))

        x = self.x(_s,_log(' - x',k))

        _y = _or(u*x,b)
        _l = x * _y
        l = self.l(_l,_log(' - l',k))

        y[k] = self.y(_y,_log(' - y',k))
        return y

    def split(self,y,log=None):        # split y-vector into context and feedforward
        nc,nf = self.sizes;
        c = y[:nc];  f = y[nc:nc+nf]
        if log is not None:
           print("\nf:",f,", c:",c)
        return (c,f)

    def __repr__(self):
        #state = "), <updy> = <%g%g%g%g>" % (self.u,self.p,self.d,self.y)
        name = self.name if self.name is not None else ''
        return "Neurotron('%s',%g)"% (name,self.k) + ""

#===========================================================================
# class Record
#===========================================================================

class Record:
    def __init__(self,cells):
        self.n = len(cells.cells)
        self.clear()

    def clear(self):                        # clear recorder
        n = self.n
        self.u = [[] for k in range(n)];
        self.q = [[] for k in range(n)];
        self.x = [[] for k in range(n)];
        self.l = [[] for k in range(n)];
        self.b = [[] for k in range(n)];
        self.d = [[] for k in range(n)];
        self.y = [[] for k in range(n)];

    def __call__(self,cells):               # record state of cells
        for k in cells.range():
            self.u[k].append(cells[k].u.out())
            self.q[k].append(cells[k].q.out())
            self.x[k].append(cells[k].x.out())
            self.l[k].append(cells[k].l.out())
            self.b[k].append(cells[k].b.out())
            self.d[k].append(cells[k].d.out())
            self.y[k].append(cells[k].y.out())

    def log(self,y,tag=None):
        print('\nsummary',tag)
        print("   u:",self.u)
        print("   q:",self.q)
        print("   x:",self.x)
        print("   l:",self.l)
        print("   b:",self.b)
        print("   d:",self.d)
        print("   y:",self.y)
        nc,nf = self[0].sizes
        print("y = [c,f]:",[y[:nc],y[nc:nc+nf]])

    def pattern(self):
        m = len(self.u);  n = len(self.u[0])
        str = '';
        for i in range(m):
            line = '';  sep = ''
            for j in range(n):
                chunk = ''
                if self.u[i][j]: chunk += 'U'
                if self.q[i][j]: chunk += 'Q'
                if self.x[i][j]: chunk += 'X'
                if self.l[i][j]: chunk += 'L'
                if self.d[i][j]: chunk += 'D'
                if self.b[i][j]: chunk += 'B'
                if self.y[i][j]: chunk += 'Y'
                if chunk == '':
                    line += '-'
                else:
                    line += sep + chunk;  sep = ','
            str += '|' + line;
        return str + '|'

#=========================================================================
# helper: concatenate two Neurotron output lists
#=========================================================================

def cat(c,f):
    """
    cat(): concatenate two Neurotron output lists, return also sizes
    >>> c = [0,0,0];  f = [0,1,0,1,0,1,0,1,0,1]
    >>> y,sizes = cat(c,f)
    >>> (y,sizes)
    ([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], (3, 10))
    """
    sizes = (len(c),len(f))
    return (c+f,sizes)

#===============================================================================
# unit test cases:
#===============================================================================

def _case1():
    """
    >>> P = Pulse(0,0)
    >>> for k in range(3): y = P(k<1,'')
      1 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
    """

def _case2():
    """
    >>> P = Pulse(1,0)
    >>> for k in range(3): y = P(k<1,'')
      1 -> (1,L1,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
    """

def _case3():
    """
    >>> P = Pulse(1,0)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (1,L1,[1,0,0]) -> 0
      1 -> (1,L1,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
    """

def _case4():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (1,L1,[1,1,0]) -> 0
      1 -> (1,D1,[1,1,0]) -> 1
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
    """

def _case5():
    """
    >>> P = Pulse(0,2)
    >>> for k in range(4): y = P(k<1,'')
      1 -> (0,D2,[0,2,0]) -> 1
      0 -> (0,D1,[0,2,0]) -> 1
      0 -> (0,L0,[0,2,0]) -> 0
      0 -> (0,L0,[0,2,0]) -> 0
    """

def _case6():
    """
    >>> P = Pulse(2,3)
    >>> for k in range(7): y = P(k<3,'')
      1 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      1 -> (2,D3,[2,3,0]) -> 1
      0 -> (1,D2,[2,3,0]) -> 1
      0 -> (0,D1,[2,3,0]) -> 1
      0 -> (0,L0,[2,3,0]) -> 0
      0 -> (0,L0,[2,3,0]) -> 0
    """

def _case7():
    """
    >>> P = Pulse(2,3)
    >>> for k in range(7): y = P(k<5 and k!=2,'')
      1 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      0 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      1 -> (2,D3,[2,3,0]) -> 1
      0 -> (1,D2,[2,3,0]) -> 1
      0 -> (0,D1,[2,3,0]) -> 1
    """

def _case8():
    """
    >>> P = Pulse(0,1,3)
    >>> for k in range(8): y = P(k<5,'')
      1 -> (0,D1,[0,1,3]) -> 1
      1 -> (0,R3,[0,1,3]) -> 0
      1 -> (0,R2,[0,1,3]) -> 0
      1 -> (0,R1,[0,1,3]) -> 0
      1 -> (1,L1,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
    """

def _case9():
    """
    >>> P = Pulse(1,1,3)
    >>> for k in range(8): y = P(k<5,'')
      1 -> (1,L1,[1,1,3]) -> 0
      1 -> (1,D1,[1,1,3]) -> 1
      1 -> (1,R3,[1,1,3]) -> 0
      1 -> (1,R2,[1,1,3]) -> 0
      1 -> (1,R1,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
    """

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
