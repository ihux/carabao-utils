#===============================================================================
# carabao/neuron package: copyright: Neuronycs 2023
# - class Synapses
# - class Rules
# - class Cell
#===============================================================================

import numpy
from numpy import arange, copy, array, transpose
from ypstruct import struct
from carabao.util import column, sat, repr

#===============================================================================
# class Synapses
#===============================================================================

class Synapses:
    """
    Synapses: class
        syn = Synapses(g)              # construct synaptic field
        syn = Synapses(K)              # construct synaptic bank

        syn.parameter(eta,theta,(plus,minus))  # setup synaptic parameters
        syn.parameter()                # eta=0.5, theta=2, plus=minus=0.02

        K = syn.K                      # synaptic index matrix
        eta = syn.eta                  # synaptic threshold
        theta = syn.theta              # spiking threshold
        plus,minus = syn.delta         # learning deltas

        v = syn.v(c)                   # group activation
        s = syn.s(c,P)                 # spike vector

        V = syn.V(c)                   # presynaptic signals
        W = syn.W(P)                   # synaptic weight matrix
        E = syn.E(c,P)                 # empowering matrix
        S = syn.S(c,P)                 # spike matrix (learning mask)
        L = syn.L(c,S)                 # learning matrix (deltas)

        P = syn.sat(P)                 # truncate P matrix to range [0,1]
    """
    def __init__(self,K):
        self.K = array(K)              # always store as numpy array
        self.parameter()               # setup default parameters

    def parameter(self,eta=0.5,theta=2,delta=(0.02,0.02)):
        self.eta = eta                 # synaptic threshold
        self.theta = theta             # spiking threshold
        self.delta = delta             # learning delta (plus,minus)

    def v(self,c):                     # group activation
        v = [c[k] if k < len(c) else 0 for k in self.K]
        return array(v)

    def s(self,c,P):                   # group activation
        S = self.S(c,P)
        return array([S[k].max() for k in S])

    def V(self,c):                     # pre-synaptic signals V(c;K)
        kmax = len(c)
        V = 0*self.K
        for mu in range(0,self.K.shape[0]):
            for nu in range(0,self.K.shape[1]):
                k = self.K[mu,nu]
                V[mu,nu] = c[k] if k < kmax else 0;
        return V

    def W(self,P):                     # binary weights W(P)
        return (P >= self.eta)*1

    def E(self,c,P):                   # E(c,P) = V(c) * W(P)
        V = self.V(c)
        return V * self.W(P)           # empowerment matrix

    def S(self,c,P):
        E = self.E(c,P);
        zero = 0 * E[0];  rng = range(0,E.shape[0])
        return array([zero + (sum(E[i])>=self.theta) for i in rng]);

    def L(self,c,S):                 # learning matrix
        V = self.V(c);
        plus,minus = self.delta
        return (2*plus * V - minus) * S

    def sat(self,X):  # truncates every matrix element of X to range 0.0 ... 1.0
        def lt1(X): return 1 + (X-1<=0)*(X-1)
        def gt0(X): return (X>=0)*X
        return lt1(gt0(X))

#===============================================================================
# class Rules
#===============================================================================

class Rules:
    """
    class Rules: provides the following rules:
        rule 0: a burst state is transient
        rule 1: excited predictive cells get active
        rule 2: excited neurons in non-predictive groups burst
        rule 3: excited bursting neurons get active
        rule 4: empowered dendritic segments spike
        rule 5: spiking dentrites of active neurons learn
        rule 6: spiking neurons get always predictive

    To apply rule:
        rules = Rules()
        c = rules.rule1(cell,u,c)
    """

    def __init__(self):
        return

    def rule0(self,cell,u,c):   # a burst state is transient
        cell.b = 0                     # clear burst state
        return cell.update(u,c,0)      # P=P', L=L', b=0

    def rule1(self,cell,u,c):   # excited predictive cells get active
        cell.y = u * cell.x
        return cell.update(u,c,1)      # y = u*x

    def rule2(self,cell,u,c):   # excited neurons in non-predictive groups burst
        v = cell.group.v(c)            # the group's outputs
        cell.b = u * (sum(v) == 0)     # set cell's burst state
        return cell.update(u,c,2)      # b = u*(sum(v)==0)

    def rule3(self,cell,u,c):   # excited bursting neurons get active
        cell.y = u * (cell.x or cell.b)
        return cell.update(u,c,3)      # y = u*(x|b)

    def rule4(self,cell,u,c):   # active predictive neurons learn
        cell._P = cell.syn.sat(cell.P+cell.y*cell.L)  # adapt permanences
        return cell.update(u,c,4)      # P' = sat(P+y*L)

    def rule5(self,cell,u,c):   # spiking neurons get always predictive
        S = cell.syn.S(c,cell.P)
        cell.x = S.max()               # dendritic spikes set cell predictive
        return cell.update(u,c,5)      # x = max(S(c,P))

    def rule6(self,cell,u,c):   # spiking dendritic segments potentially learn
        S = cell.syn.S(c,cell.P)
        cell._L = cell.syn.L(c,S)      # learning deltas
        return cell.update(u,c,6)      # L' = L(c,S(c,P))



#=============================================================================
# class Cell
#=============================================================================

class Cell:
    """
    Cell: class Cell - modelling HTM cell algorithm

        from carabao.screen import Monitor
        from carabao.cell import Cell,toy
        from numpy import ones

        mon = Monitor(m=1,n=3)
        k,g,K,P,c = toy('cell')
        cell = Cell(mon,k,g,K,P)
        cell.plot(i=0,j=0)    # plot at monitor location i,j

        cell.u = cell.y = cell.x = cell.b = 1
        cell.plot(i=0,j=1)    # plot at monitor location i,j

        v = [1,0,1,0];  V = ones((2,5));  E = (cell.P >= 0.5)*V
        cell.plot(0,2,v,E)

        see also: Synapses, Rules
    """

    def __init__(self,mon,k,g,K,P):
        self.mon = mon.copy()  # Monitor(mon.screen.m,mon.screen.n,mon.verbose)
        self.rules = Rules()   # add a rules object

        self.k = k;
        self.group = Synapses(g)          # synaptic field (group activity))
        self.syn = Synapses(K)            # synaptic bank (any context activity)

        self._u = 0                       # basal (feedforwad) input
        self._c = []                      # context input

        self.y = 0                        # cell output (axon)
        self.x = 0                        # predictive state
        self.b = 0                        # burst state
        self.P = P                        # permanence matrix
        self.L = 0*P.copy()               # pre-synaptic pattern

        self._P = self.P.copy()           # new permanences @ t+1
        self._L = self.L.copy()           # new learning matrix @ t+1

    def rule0(self,u,c): return self.rules.rule0(self,u,c)
    def rule1(self,u,c): return self.rules.rule1(self,u,c)
    def rule2(self,u,c): return self.rules.rule2(self,u,c)
    def rule3(self,u,c): return self.rules.rule3(self,u,c)
    def rule4(self,u,c): return self.rules.rule4(self,u,c)
    def rule5(self,u,c): return self.rules.rule5(self,u,c)
    def rule6(self,u,c): return self.rules.rule6(self,u,c)

    def transition(self):
        self.P = self._P.copy()        # permanence matrix transition
        self.L = self._L.copy()        # learning matrix transition

    def phase(self,ph,u,c):            # cell algo phase `ph`
        if ph == 1:
            c = self.rule0(u,c) # a burst state is transient
            c = self.rule1(u,c) # excited predictive cells get active
        elif ph == 2:
            c = self.rule2(u,c) # excited neurons in non-predictive groups burst
        elif ph == 3:
            c = self.rule3(u,c) # excited bursting neurons get active
            c = self.rule4(u,c) # spiking dentrites of active neurons learn
        elif ph == 4:
            c = self.rule5(u,c) # empowered dendritic segments spike
            c = self.rule6(u,c) # spiking neurons get always predictive
            self.transition()
        else:
            raise Exception("bad phase")
        return c

    def update(self,u,c,phase):        # update context with current output
        self.set(u=u,c=c)              # store for plot routines
        c = c.copy();                  # update a copy of the context
        while len(c) <= self.k: c.append(0)
        c[self.k] = self.y             # with changed output
        return c

    def log(self,txt):
        self.mon.log(self,txt)

    def plot(self,i=None,j=None,v=None,W=None,E=None,u=None,c=None,
	         xlabel=None,head=None,foot=None):
	    self._u = self._u if u is None else u
	    self._c = self._c if c is None else c
	    self.mon.plot(self,i,j,v=v,W=W,E=E,u=u,c=c)
	    if xlabel is not None: self.mon.xlabel(j,xlabel)
	    if head is not None: self.mon.head(head)
	    if foot is not None: self.mon.foot(foot)

    def set(self,u=None,c=None,x=None,y=None,b=None):
        self._u = self._u if u is None else u
        self._c = self._c if c is None else c
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.b = self.b if b is None else b
        return self


#===============================================================================
# helper: toy cell parameters
#===============================================================================

def toy(tag):
    """
    Toy: create a toy object

       k,g,K,P,c = toy('cell')
       k,g,K,P,c = toy('mini3')
       G,K,P,c,token,xlabel,minicol = toy('tiny')
    """
    if tag == 'cell':
        k = 0                        # cell index
        g = [0,1,2,3]                # group indices
        K = array([[1,3,5,7,9],[5,6,7,8,9]])
        P = array([[0.12,0.32,0.54,0.77,0],[0,0.61,0.45,0,0.8]])
        c = [0,0,0,0,1,1,1,1,1,0];
        return k,g,K,P,c
    elif tag == 'mini3':
        k = [0,1,2]                  # cell indices
        g = [0,1,2]                  # group indices
        K0 = array([[3,4,5,6,7],[1,3,5,7,9]])
        K1 = array([[4,5,6,7,8],[2,4,5,6,7]])
        K2 = array([[5,6,7,8,9],[0,3,7,8,9]])
        P0 = array([[0.5,0.6,0.1,0.2,0.3],[0.0,0.6,0.4,0.0,0.0]])
        P1 = array([[0.1,0.3,0.5,0.6,0.0],[0.0,0.5,0.5,0.7,0.0]])
        P2 = array([[0.0,0.1,0.5,0.7,0.1],[0.0,0.1,0.3,0.8,0.0]])
        K = [K0,K1,K2]
        P = [P0,P1,P2]
        c = [0,0,0,0,1,1,0,1,1,0];
        return k,g,K,P,c
    elif tag == 'tiny':
        m = 2;  n = 7
        K = array([[array([[0,1,2,3,4],[5,6,7,8,9]])
                    for j in range(0,n)] for i in range(0,m)])
        P = array([[array([[0,0,0,0,0],[0,0,0,0,0]])
            for j in range(0,n)] for i in range(0,m)])
        G = transpose(array([k for k in range(0,m*n)]).reshape(m,n,order='F'))
        c = [0,0,0,0, 0,0,0, 0,0,0,0, 0,0,0]
        token = {'Mary':[1,0,0,0,0,0,1], 'John':[0,1,0,0,0,0,1],
                 'likes':[0,0,1,0,0,0,1], 'to':[0,0,0,1,0,0,1],
                 'sing':[0,0,0,0,1,0,1], 'dance':[0,0,0,0,0,1,1]}
        xlabel = ['Mary','John','likes','to','sing','dance','X']
        minicol = {'Mary':0, 'John':1, 'likes':2, 'to':3, 'sing':4, 'dance':5}
        return G,K,P,c,token,xlabel,minicol
    else:
        raise Exception('unknown tag')

def hello():
	print("hello, world!")
