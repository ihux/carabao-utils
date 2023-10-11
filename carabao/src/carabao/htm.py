#============================================================================
# HTM Classes
# usage: lay = Layer(4,10,5,2)
#        lay.input(token)
#        lay.print()
#============================================================================

from carabao.util import peek,poke,Random
from carabao.toy import Toy
from torch import zeros

#===============================================================================
# class: HTM layer
#===============================================================================

  # Layer(m,n,s,d)
  #   m: number of cells per minicolumn
  #   n: number of minicolumns
  #   s: number of synapses per distal segment
  #   d: number of distal segments

import torch

class Layer:
  from torch import zeros, rand
  def input(self,word,vocabulary):
    self.u = vocabulary[word]

  def __init__(self, m, n, s, d):
    self.m = m;  self.n = n;  self.s = s;  self.d = d;
    self.N = m*n;  self.M = s * d
    rand = Random(0)                    # create random generator @ seed = 0
    self.X = torch.zeros(m,n)
    self.Y = self.X
    self.u = None
    M0 = self.forbidden(self.s,self.d)
    P0 = torch.zeros(self.m,self.n)
    self.K = rand.cells(P0,M0,self.N)
    self.P = torch.rand(self.m,self.n,self.s,self.d)

  def forbidden(self,m,n):              # setup forbidden matrix
    M = torch.zeros(m,n)
    for i in range(0,m):
      for j in range(0,n):
        M[i,j] = j*n + i
    return M

  def input(self,u):
    self.u = u;

  def print(self):
    print("Layer ",self.m,"x",self.n,"cells @ ",self.s,"x",self.d,"synapses")
    print("u:\n",self.u)
    print("X:\n",self.X)
    print("Y:\n",self.Y)
    print("K:\n",self.K)
    print("K:\n",self.K)
