#===============================================================================
# carabao/rcell package: copyright: Neuronycs 2023
# - class Rcell:   rule based HTM Neuron class
# - toy()          quick setup of cell parameter matrices
# - select():      implementing context selection by index matrix
# - norm1():       calculate 1-norm of a matrix
# - sat():         saturate a matrix to elements in [0,1]
# - column():      create mx1 column matrix from list or array
# - ones():        setup matrix with all ones
# - zeros():       setup matrix with all zeros
#===============================================================================

import numpy
from numpy import transpose as trn
from numpy import ones, arange, copy, array
from ypstruct import struct

#=============================================================================
# select function
#=============================================================================

def select(K,c):
    """
    select(): select a set of neuron outputs given an index matrix/list and
              a context vector c

                  c = [0,0,0,1,1, 1,1,0,0,0]
                  g = [0,1,2];
                  v = select(g,c)

                  K = numpy.array([[4,5,6,7],[6,7,8,9]])
                  V = select(K,c)

              see also: Cell, Rcell, hash, norm1
    """
    if type(K).__name__ == 'list':
        return [c[k] for k in K];

    if type(K).__name__ != 'ndarray':
        raise Exception('numpy array expected')

    V = copy(K);       # make a copy
    for i in range(0,K.shape[0]):
        for j in range(0,K.shape[1]):
            V[i,j] = c[K[i,j]]

    return V

#===============================================================================
# helper: vector 1-norm or matrix 1-norm
#===============================================================================

def norm1(M):    # max of row sums
    """
    norm1(): vector 1-norm or matrix 1-norm

                 v = [2,-3,-1]      # list representation of vector
                 n = norm1(v)       # sum of abs values => n = 6

                 V = numpy.array([[2,-3,-1],[1,0,-1]])
                 n = norm1(V)       # max of row 1-norm => n = max(6,2) =6

             see also: Cell, select, hash
    """
    if type(M).__name__ == 'list':
        return sum(M)

    result = 0
    for j in range(0,M.shape[0]):
        sumj = M[j].sum().item()
        result = result if sumj < result else sumj
    return result

#===============================================================================
# helper: sat function fopr a numpx matrix
# - truncates every matrix element to range 0.0 ... 1.0
#===============================================================================

def sat(X):
    def lt1(X): return 1 + (X-1<=0)*(X-1)
    def gt0(X): return (X>=0)*X
    return lt1(gt0(X))

#=============================================================================
# helper: create column vector from list
#=============================================================================

def column(x):
    """
    column(): create column vector from list

        v = column([0,1])
    """
    return trn(array([x]))

#=============================================================================
# helper: zeros() and ones()
#=============================================================================

def ones(arg):
    if type(arg).__name__=='tuple':
        return numpy.ones(arg)
    else:
        one = numpy.ones(arg.shape)
        if len(one.shape) == 1:
            one = array([one])
        return one

def zeros(arg):
    if type(arg).__name__=='tuple':
        return numpy.zeros(arg)
    else:
        zero = numpy.zeros(arg.shape)
        if len(zero.shape) == 1:
            zero = array([zero])
        return zero

#=============================================================================
# class Cell
#=============================================================================

class Rcell:
    """
    Rcell: class Rcell - modelling HTM cell algorithm

        from carabao.screen import Monitor
        from carabao.cell import Cell,toy
        from numpy import ones

        mon = Monitor(m=1,n=3)
        k,g,K,P,c = toy('cell')
        cell = Rcell(mon,k,g,K,P)
        cell.plot(i=0,j=0)    # plot at monitor location i,j

        cell.u = cell.y = cell.x = cell.b = 1
        cell.plot(i=0,j=1)    # plot at monitor location i,j

        v = [1,0,1,0];  V = ones((2,5));  E = (cell.P >= 0.5)*V
        cell.plot(0,2,v,E)
    """
    def __init__(self,mon,k,g,K,P):
        self.mon = mon.copy()  # Monitor(mon.screen.m,mon.screen.n,mon.verbose)
        zero = [0 for i in range(0,P.shape[0])]

            # input, output, state variables

        self.y = 0                     # cell output (axon)
        self.x = 0                     # predictive state
        self.b = 0                     # burst state
        #self.s = 0*P[:,0]             # zero spike state
        self.P = P                     # permanence matrix (state)

            # auxiliary quantities

        self.aux = struct()
        self.aux.u = 0                 # basal (feedforwad) input
        self.aux.c = []                # context input
        #self.aux.v = [0,0,0,0]         # group outputs
        self.aux.V = 0*P               # pre-synaptic signals
        self.aux.W = 0*P               # dendritic weights
        self.aux.E = 0*P               # empowerment matrix
        self.aux.L = 0*P               # learning mask
        self.aux.D = 0*P               # learning delta

            # parameters and auxilliary variables

        self.config(k,g,K)
        self.x_ = 0                    # auxilliary: x(t+1)
        self.P_ = self.P               # auxilliary: P(t+1)

    def config(self,k,g,K):
        self.eta = 0.5                 # synaptic threshold
        self.theta = 2                 # dendritic threshold
        self.pdelta = 0.04             # positive learning delata
        self.ndelta = 0.02             # negative learning delta
        self.k = k;
        self.g = g;
        self.K = K;

    def v(self,c):
        if c == []: return 0*array(self.g)
        return array([c[k] for k in self.g])   # group output

    def V(self,c):                    # pre-synaptic signals
        if c == []: return 0*self.P
        V = 0*self.K
        for mu in range(0,self.K.shape[0]):
            for nu in range(0,self.K.shape[1]):
                V[mu,nu] = c[self.K[mu,nu]];
        return V

    def W(self):
        return (self.P >= self.eta)*1  # synaptic (binary) weights

    def E(self,c):
        return self.V(c) * self.W()    # empowerment matrix

    def L(self,c):
        return column(self.s(c)) @ ones(self.P[0,:])

    def s(self,c):                     # spike vector
        E = self.E(c)
        _s = [(sum(E[mu]) >= self.theta)
             for mu in range(0,E.shape[0])]
        return 1*array(_s)

    def update(self,u,c,args):
        self.aux.u = u                 # store for analysis
        self.aux.c = c                 # store for analysis
        c[self.k] = self.y             # update context with changed output
        if args['rule'] == 2:
            self.aux.v = args['v']
        if args['rule'] == 4:
            self.aux.V = args['V']
            self.aux.W = args['W']
            self.aux.E = args['E']
        if args['rule'] == 5:
            self.aux.L = args['L']
            self.aux.D = args['D']

        return c

      # === rule 1: excited predictive cells get active ===

    def rule1(self,u,c):
        self.y = u * self.x            # excited & predictive cells get active
        return self.update(u,c,{'rule':1})

      # === rule 2: excited neurons in non-predictive groups burst

    def rule2(self,u,c):
        v = self.v(c)                  # the group's outputs
        self.b = u * (sum(v) == 0)     # set cell's burst state

           # important: don't change output (and context vector) in this phase
           # before all cells in the context have determined their burst state

        return self.update(u,c,{'rule':2, 'v':v})

      # === rule 3: excited bursting neurons get active ===

    def rule3(self,u,c):
        self.y = u * (self.y or self.b)
        return self.update(u,c,{'rule':3})

      # === rule 4: empowered dendritic segments spike ===

    def rule4(self,u,c):
        V = self.V(c)                            # pre-synaptic signals
        W = self.W()                             # synaptic weights
        E = V * W                                # empowerment matrix
        #s = u * self.s(c)                       # spike vector
        s = self.s(c)                            # spike vector

        return self.update(u,c,{'rule':4, 'V':V, 'W':W, 'E':E})

       # === rule 5: spiking dentrites of active cells learn

    def rule5(self,u,c):
        V = self.V(c)                            # pre-synaptic signals
        L = self.L(c)
        D = L*(self.pdelta * V - self.ndelta)
        self.P = sat(self.P + self.y * D)        # learning (adapt permanences)
        return self.update(u,c,{'rule':5, 'L':L, 'D':D})

      # === rule 6: spiking neurons get always predictive ===

    def rule6(self,u,c):
        self.x = max(self.s(c))       # dendritic spikes set cells predictive
        return self.update(u,c,{'rule':6})

    def log(self,txt):
        self.c = self.aux.c
        self.mon.log(self,txt)

    def plot(self,i=None,j=None,v=None,W=None,E=None,xlabel=None,head=None,foot=None):
        self.u = self.aux.u # for compatibility
        #v = self.aux.v if v is None else v
        #V = self.aux.V if V is None else V
        W = self.aux.W if W is None else W
        E = self.aux.E if E is None else E

        self.mon.plot(self,i,j,v,W,E)
        if xlabel is not None:
            self.mon.xlabel(j,xlabel)
        if head is not None:
            self.mon.head(head)
        if foot is not None:
            self.mon.foot(foot)

    def set(self,u=None,c=None,x=None,y=None,b=None,s=None,v=None,V=None,W=None,E=None):
        self.aux.u = self.aux.u if u is None else u
        self.c = self.aux.c if c is None else c
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.b = self.b if b is None else b
        #self.s = self.s if s is None else s
        #self.aux.v = self.aux.v if v is None else v
        self.aux.V = self.aux.V if V is None else V
        self.aux.W = self.aux.W if W is None else W
        self.aux.E = self.aux.E if E is None else E
        return self

#===============================================================================
# class Toy
# - usage: k,g,P,K = Toy('cell')
#===============================================================================

def toy(tag):
    """
    Toy: create a toy object

       k,g,K,P,c = toy('cell')
       k,g,K,P,c = toy('mini3')
    """
    if tag == 'cell':
        k = 0                        # cell index
        g = [0,1,2,4]                  # group indices
        K = array([[1,3,5,7,9],[3,4,5,6,7]])
        P = array([[0.12,0.32,0.54,0.77,0],[0,0.61,0.45,0,0]])
        c = [0,0,0,0,1,1,0,1,1,0];
        return k,g,K,P,c
    elif tag == 'mini3':
        k = [0,1,2]                  # cell indices
        g = [0,1,2]                  # group indices
        K0 = array([[3,4,5,6,7],[1,3,5,7,9]])
        K1 = array([[4,5,6,7,8],[2,4,5,6,7]])
        K2 = array([[5,6,7,8,9],[0,3,7,8,9]])
        P0 = array([[0.5,0.6,0.1,0.2,0.3],[0.0,0.6,0.4,0.0,0.0]])
        P1 = array([[0.1,0.3,0.5,0.6,0.0],[0.0,0.6,0.5,0.7,0.0]])
        P2 = array([[0.0,0.1,0.5,0.7,0.1],[0.0,0.1,0.3,0.8,0.0]])
        K = [K0,K1,K2]
        P = [P0,P1,P2]
        c = [0,0,0,0,1,1,0,1,1,0];
        return k,g,K,P,c
    else:
        raise Exception('unknown tag')
