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
            raise Exception()
        elif state == 'R':              # transition to relax state
            self.c = r                  # set relax counter
        self.s = state                  # actual state change

    def integrate(self,u):
        self.u = u;
        l = self.n[0]                   # lag duration
        i = self.x + 2*u - 1            # integrator output
        self.x = max(0,min(i,l))        # limit integrator state
        return self.x,i                 # return integrator state/output

    def call(self,u):
        l,d,r = self.n                  # get parameters
        x,i = self.integrate(u)         # integrate
        y_ = int(i > l and d >0)        # proposed new output
        c_ = d if x >= l > 0 and r == 0 else self.c - 1  # proposed new count

        if self.s == 'L':               # L: lag state (debouncing)
            if y_ > 0:
                self.c = d;  self.y = y_
                self.s = 'D'
            else:
                self.c = x;  self.y = y_
        elif self.s == 'D':             # D: duty state
            if self.c <= 1 and r > 0:
                self.y = 0;  self.c = r
                self.trans('R')         # transition to relax state
            #elif l == 0 and u > 0:
            #    self.y = y_
            elif c_ <= 0 and r == 0:
                self.y = 0;  self.c = u
                self.trans('L')         # transition to lag state
            else:
                self.c = c_;
                self.y = int(c_ > 0)
        elif self.s == 'R':             # R: relax state
            if self.c <= 1 and l == 0 and u > 0:
                self.c = d;  self.y = y_
                self.s = 'D'
            elif self.c <= 1:
                self.c = u              # count down relax period
                self.trans('L')
            else:
                self.c -= 1             # count down relax period

        if self.name is not None: print(self)
        return self.out()

    def inp(self): return self.u
    def out(self): return self.y
    def set(self,val,log=None):
        if val > 0:
            self.c = self.n[1];  self.s = 'D'
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
        s = cell.s.out()
        self.screen.neurotron((i,j),u,q,x,y,b,d,l,s)
    def xlabel(self,x,txt,size=None):
        self.screen.text(x,-0.75,txt,size=size)
    def title(self,txt,size=10):
        scr = self.screen
        x = (scr.n-1)/2;  y = scr.m + 0.3
        self.screen.text(x,y,txt,size=size)

#===============================================================================
# class Neurotron
#===============================================================================

class Neurotron:
    """
    class Neurotron: full functionality
    >>> par,sizes = toy('sarah') # epar,dpar,ppar,dyn = par
    >>> cell0 = Neurotron(k:=0,par,sizes,'cell0')
    >>> print(cell0)
    Neurotron('cell0',0)
    """
    def __init__(self,k,par,sizes,name=None):
        self.k = k
        self.sizes = sizes
        self.name = name

        epar,dpar,ppar,dyn = par

        self.excite  = Terminal(epar.w[k],epar.theta,'excite')
        self.excite.synapses = Synapses(epar.k[k],epar.p[k],epar.eta)

        self.depress = Terminal(dpar.w[k],dpar.theta,'depress')
        self.depress.synapses = Synapses(dpar.g[k],dpar.p[k],dpar.eta)

        self.predict = Terminal(ppar.W[k],ppar.theta,'predict')
        self.predict.synapses = Synapses(ppar.K[k],ppar.P[k],ppar.eta)

        self.u = Pulse(*dyn['u'])
        self.q = Pulse(*dyn['q'])
        self.x = Pulse(*dyn['x'])
        self.y = Pulse(*dyn['y'])
        self.d = Pulse(*dyn['d'])
        self.b = Pulse(*dyn['b'])
        self.l = Pulse(*dyn['l'])
        self.s = Pulse(*dyn['s'])

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
        s = self.s(_s,_log(' - s',k))

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
        self.s = [[] for k in range(n)];

    def __call__(self,cells):               # record state of cells
        for k in cells.range():
            self.u[k].append(cells[k].u.out())
            self.q[k].append(cells[k].q.out())
            self.x[k].append(cells[k].x.out())
            self.l[k].append(cells[k].l.out())
            self.b[k].append(cells[k].b.out())
            self.d[k].append(cells[k].d.out())
            self.y[k].append(cells[k].y.out())
            self.s[k].append(cells[k].s.out())

    def log(self,cells,y,tag=None):
        print('\nSummary:',tag)
        print("   u:",self.u)
        print("   q:",self.q)
        print("   x:",self.x)
        print("   l:",self.l)
        print("   b:",self.b)
        print("   d:",self.d)
        print("   y:",self.y)
        print("   s:",self.s)
        nc,nf = cells[0].sizes
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
                if self.s[i][j]: chunk += 'S'
                if chunk == '':
                    line += '-'
                else:
                    line += sep + chunk;  sep = ','
            str += '|' + line;
        return str + '|'

#===============================================================================
# helper: set up toy stuff
#===============================================================================

def toy(mode):
    """
    toy(): setup toy stuff
    >>> par,token = toy('sarah') # get params for 'sarah' app
    >>> excite,depress,predict,dyn = par
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

        dyn = {'u':(0,4,3), 'q':(2,1,0), 'x':(1,7,0), 'y':(1,2,0),
               'd':(0,2,0), 'b':(0,2,2), 'l':(1,1,5), 's':(0,1,6)}
        #dyn = {'u':(0,4,3), 'q':(2,1,0), 'x':(1,7,0), 'y':(1,2,0),
        #       'd':(0,2,0), 'b':(0,1,0), 'l':(1,1,5)}

        return (e,d,p,dyn),token

    elif mode == 'tony':
        par,token = toy('sarah')
        token = {'Tony':[1,1,0,1,1,1,0,1,0,1],
                 'loves':[0,1,1,1,0,1,1,0,1,1],
                 'cars':[1,1,1,0,0,1,0,1,1,1]}
        e,d,p,dyn = par
        dyn = {'u':(2,4,4), 'q':(2,1,0), 'x':(1,9,0), 'y':(1,2,0),
               'd':(0,2,0), 'b':(0,2,2), 'l':(1,1,5), 's':(0,1,6)}
        return (e,d,p,dyn),token


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

def _case1a():
    """
    >>> P = Pulse(0,0)
    >>> for k in range(3): y = P(k<1,'')
      1 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
    """

def _case1b():                  # Identity
    """
    >>> P = Pulse(0,1)
    >>> for k in range(6): y = P(k<1 or 3 <=k <= 4,'')
      1 -> (0,D1,[0,1,0]) -> 1
      0 -> (0,L0,[0,1,0]) -> 0
      0 -> (0,L0,[0,1,0]) -> 0
      1 -> (0,D1,[0,1,0]) -> 1
      1 -> (1,L1,[0,1,0]) -> 0
      0 -> (0,L0,[0,1,0]) -> 0
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

def _case4a():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(k<1,'')
      1 -> (1,L1,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
    """

def _case4b():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (1,L1,[1,1,0]) -> 0
      1 -> (1,D1,[1,1,0]) -> 1
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
    """

def _case4c():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(1,'')
      1 -> (1,L1,[1,1,0]) -> 0
      1 -> (1,D1,[1,1,0]) -> 1
      1 -> (1,D1,[1,1,0]) -> 1
      1 -> (1,D1,[1,1,0]) -> 1
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
    >>> for k in range(8): y = P(k<5 and k!=2,'')
      1 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      0 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      1 -> (2,D3,[2,3,0]) -> 1
      0 -> (1,D2,[2,3,0]) -> 1
      0 -> (0,D1,[2,3,0]) -> 1
      0 -> (0,L0,[2,3,0]) -> 0
    """

def _case8():
    """
    >>> P = Pulse(0,1,3)
    >>> for k in range(8): y = P(k<4,'')
      1 -> (0,D1,[0,1,3]) -> 1
      1 -> (0,R3,[0,1,3]) -> 0
      1 -> (0,R2,[0,1,3]) -> 0
      1 -> (0,R1,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
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
