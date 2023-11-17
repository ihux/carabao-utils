#===============================================================================
# carabao.neurotron package: copyright: Neuronycs 2023
#===============================================================================
"""
Module carabao.neurotron supports the following classes:
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

from numpy import array, transpose
from ypstruct import struct
from carabao.util import repr
from carabao.screen import Screen
from carabao.pulse import Pulse

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
        self.L = self.P*0              # learning delta
        self.eta = eta                 # synaptic threshold
        self.log = log                 # log header (no logging if log=None)
        self.delta = (0.1,0.1)         # learning delta

    def weight(self):
        W = (self.P >= self.eta)*1;
        return W

    def __call__(self,y,log=None):     # feed network state to synapses
        eta = self.eta;  K = self.K;  V = 0*K;
        W = self.weight()
        for i in range(0,K.shape[0]):
            for j in range(0,K.shape[1]):
                V[i,j] = y[K[i,j]] if W[i,j] > 0 else 0
        if self.log is not None:
            print(self.log,repr(y),"->",repr(W),"->",repr(V))
        return V

    def mind(self,V,S):                # mind a potential learning delta
        pdelta,ndelta = self.delta
        self.L = S*(2*pdelta * V - ndelta)
        #print('***** L:',self.L)

    def learn(self):
        self.P += self.L
        for i in range(self.P.shape[0]):    # limit 0 <= P <= 1
            for j in range(self.P.shape[1]):
                self.P[i,j] = max(0.0,min(self.P[i,j],1.0))
        print('=> learning: L:',repr(self.L))
        print('             P:',repr(self.P))

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
        self.name = name            # name string
        self._s = None              # to save s
        self._V = None              # to save V
        #if name is not None:
        #    print(self)

    def empower(self,V,log=None):      # determine empowerment
        if self.synapses is not None:
            self.W = self.synapses.weight()
        E = self.W * V
        #print("   ***** W:\n",self.W)
        #print("   ***** V:\n",V)
        #print("   ***** E:\n",E)
        if log is not None:
            print(log,repr(V),"->",repr(E))
        return E

    def spike(self,E,log=None):        # spike function
        S = array([sum(E[k]) for k in range(0,E.shape[0])])
        s = (S >= self.theta)*1
        if log is not None:
            print(log,repr(E),"->",repr(s))
        return s

    def V(self,x):                     # presynaptic signals
        K = self.synapses.K
        V = 0*K;  m,n = K.shape
        #print('***** m:',m,'n:',n,'K:\n',K)
        for i in range(m):
            for j in range(n):
                V[i][j] = x[K[i][j]]
        return V

    def S(self,s):                    # spike matrix (learning mask)
        one = array([[1 for j in range(self.W.shape[1])]])
        s = transpose(array([s]))
        #print('***** ','s:',s,'one:',one)
        S = s @ one
        #print('***** S:\n',S)
        return S

    def learn(self,s,l):              # learning
        if s:
            S = self.S(self._s)
            #print('***** mind self._s:',self._s,'S:\n',S)
            self.synapses.mind(self._V,S)
        if l:
            self.synapses.learn()

    def __call__(self,x,log=None):  # feed x vector to terminal
        if self.synapses is None:
            return ([],[])
        V = self.V(x)
        E = self.empower(V)
        s = self.spike(E)
        self._s = s;  self._V = V
        #print('***** s:',s,'V:\n',V)

        if log is not None:
            if self.synapses is None:
                print(log,repr(V),"->",repr(E),"->",repr(s))
            else:
                W = self.synapses.weight()
                print(log,repr(V),"->",repr(W),"->",repr(E),"->",repr(s))
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

        self.predict.learn(s,l)        # learning

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
        d.g = bundle([0,1,2],3);               # group indices
        d.p = bundle([1,1,1],3);            # all depression permanences are 1
        d.theta = 1                         # depression threshold
        d.eta = 0.5                         # synaptic threshold

        p = struct()                        # prediction terminal parameters
        p.W = bundle([[1,0,0],[0,1,1]],3)   # prediction weights
        p.K = bundle([[0,1,2],[0,1,2]],3)
        p.P = [[prm[0:3],prm[0:3]],
               [prm[3:6],prm[0:3]],
               [prm[6:9],prm[0:3]]]
        p.theta = 1                         # prediction threshold
        p.eta = 0.5                         # synaptic threshold

        dyn = {'u':(0,4,4), 'q':(2,1,0), 'x':(1,8,0), 'y':(1,2,0),
               'd':(0,2,0), 'b':(0,2,3), 'l':(1,1,5), 's':(0,1,6)}
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
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
