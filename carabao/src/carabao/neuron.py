#===============================================================================
# carabao/neuron package: copyright: Neuronycs 2023
# - class Synapses
# - class Rules
# - class Cell
#===============================================================================

import numpy
from numpy import arange, copy, array
from ypstruct import struct
from carabao.util import column, sat

#===============================================================================
# class Synapses
#===============================================================================

class Synapses:
    """
    Synapses: class
        syn = Synapses(K,P,eta,theta,(plus,minus))   # full arg list
        syn = Synapses(K,P)       # eta=0.5, plus=minus=0.02
        syn = Synapses(g)         # P=one

        P = syn.P                 # permanences

        W = syn.W()               # synaptic weight matrix
        V = syn.V(c)              # presynaptic signals
        E = syn.E(c)              # empowering matrix
        L = syn.L(c)              # learning mask
        D = syn.D(c)              # learning deltas
        s = syn.s(c)              # spike vector
        o = syn.one               # [1,1,...,1] matrix (1 x ns)

        syn.P = syn.sat(P)        # truncate P matrix to range [0,1]
    """
    def __init__(self,K,P=None,eta=0.5,theta=2,delta=(0.02,0.02)):
        self.K = array(K)         # always store as numpy array
        self.P = array(P)         # always store as numpy array
        self.eta = eta            # synaptic threshold
        self.theta = theta        # spiking threshold
        self.delta = delta        # learning delta (plus,minus)

    def W(self):                  # binary weights
        return (self.P >= self.eta)*1

    def V(self,c):                    # pre-synaptic signals
        kmax = len(c)
        V = 0*self.K
		######################
        #print("K:",repr(self.K))
        for mu in range(0,self.K.shape[0]):
            for nu in range(0,self.K.shape[1]):
                k = self.K[mu,nu]
                V[mu,nu] = c[k] if k < kmax else 0;
        return V

    def E(self,c):                     # E = V(c) * W(P)
        return self.V(c) * self.W()    # empowerment matrix

    def S(self,c):
        E = self.E(c);
        zero = 0 * E[0];  rng = range(0,E.shape[0])
        return array([zero + (sum(E[i])>=self.theta) for i in rng]);

    def L(self,c):                     # D = (2*plus * V - minus) * L
        S = self.S(c);  V = self.V(c);
        plus,minus = self.delta
        return (2*plus * V - minus) * S

    def s(self,c):                     # spike vector: s = (sum(E')>=theta)
        S = self.S(c)
        return array([S[i].max() for i in range(0,S.shape[0])])

    def v(self,c):                     # group output
        return array([c[k] if k < len(c) else 0
                      for k in self.K])

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
        rule 1: excited predictive cells get active
        rule 2: excited neurons in non-predictive groups burst
        rule 3: excited bursting neurons get active
        rule 4: empowered dendritic segments spike
        rule 5: spiking dentrites of active neurons learn
        rule 6: spiking neurons get always predictive
        rule 7: burst states have limited duration

    To apply rule:
        rules = Rules()
        c = rules.rule1(cell,u,c)
    """

    def __init__(self):
        return

    def rule1(self,cell,u,c):   # excited predictive cells get active
        cell.y = u * cell.x
        return cell.update(u,c,1)      # update c[k] = cell.y

    def rule2(self,cell,u,c):   # excited neurons in non-predictive groups burst
        v = cell.v(c)                  # the group's outputs
        cell.b = u * (sum(v) == 0)     # set cell's burst state
        return cell.update(u,c,2)      # update c[k] = cell.y

    def rule3(self,cell,u,c):   # excited bursting neurons get active
        cell.y = u * (cell.x or cell.b)
        return cell.update(u,c,3)

    def rule4(self,cell,u,c):   # empowered dendritic segments spike
        cell.s = cell.syn.s(c)         # spike vector
        return cell.update(u,c,4)

    def rule5(self,cell,u,c):   # spiking dentrites of active neurons learn
        L = cell.L(c)                             # learning deltas
        P = cell.syn.P + cell.y * L               # adapt permanences
        cell.syn.P = cell.syn.sat(P)
        return cell.update(u,c,5)

    def rule6(self,cell,u,c):   # spiking neurons get always predictive
        cell.x = max(cell.s)           # dendritic spikes set cell predictive
        return cell.update(u,c,6)

    def rule7(self,cell,u,c):   # burst and spike states are transient
        cell.b = 0                     # clear burst state
        cell.s = 0 * cell.s            # clear spike state
        return cell.update(u,c,6)


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

            # neuron index, synaptic bank and synaptic field

        self.k = k;
        self.syn = Synapses(K,P)          # synaptic bank (any context activity)
        self.grp = Synapses(g)            # synaptic field (group activity))

            # input variables

        self.input = struct()             # structure for all neuron inputs
        self.input.u = 0                  # basal (feedforwad) input
        self.input.c = []                 # context input

            # output, state variables

        self.y = 0                        # cell output (axon)
        self.x = 0                        # predictive state
        self.b = 0                        # burst state
        self.s = self.syn.s([])           # spike vector

    def v(self,c): return self.grp.v(c)   # group output
    def s(self,c): return self.syn.s(c)   # spike vector
    def W(self):   return self.syn.W()    # synaptic weights
    def V(self,c): return self.syn.V(c)   # pre-synaptic signals
    def E(self,c): return self.syn.E(c)   # empowerment matrix
    def S(self,c): return self.syn.S(c)   # spike matrix (learning mask)
    def L(self,c): return self.syn.L(c)   # learning delta

    def rule1(self,u,c): return self.rules.rule1(self,u,c)
    def rule2(self,u,c): return self.rules.rule2(self,u,c)
    def rule3(self,u,c): return self.rules.rule3(self,u,c)
    def rule4(self,u,c): return self.rules.rule4(self,u,c)
    def rule5(self,u,c): return self.rules.rule5(self,u,c)
    def rule6(self,u,c): return self.rules.rule6(self,u,c)
    def rule7(self,u,c): return self.rules.rule7(self,u,c)

       # dynamic algorithm (comprising 3 phases)

    def phase(self,ph,u,c):            # cell algo phase i
        if ph == 1:
            c = self.rule1(u,c) # excited predictive cells get active
        elif ph == 2:
            c = self.rule2(u,c) # excited neurons in non-predictive groups burst
            c = self.rule4(u,c) # empowered dendritic segments spike
            c = self.rule5(u,c) # spiking dentrites of active neurons learn
            c = self.rule6(u,c) # spiking neurons get always predictive
        elif ph == 3:
            c = self.rule3(u,c) # excited bursting neurons get active
            c = self.rule7(u,c) # burst and spike states are transient
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
	    self.input.u = self.input.u if u is None else u
	    self.input.c = self.input.c if c is None else c
	    self.mon.plot(self,i,j,v=v,W=W,E=E,u=u,c=c)
	    if xlabel is not None: self.mon.xlabel(j,xlabel)
	    if head is not None: self.mon.head(head)
	    if foot is not None: self.mon.foot(foot)

    def set(self,u=None,c=None,x=None,y=None,b=None):
        self.input.u = self.input.u if u is None else u
        self.input.c = self.input.c if c is None else c
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
    else:
        raise Exception('unknown tag')

def hello():
	print("hello, world!")
