#===============================================================================
# carabao/util.py: carabao utilities
#===============================================================================

import numpy as np
import torch

#===============================================================================
# helper: portable pseudo random generator for random integers modulo N. Given
#         the same seed during creation, the random generator produces a repea-
#         table sequence of pseudo random numbers
# - usage: rg = Random(seed)       # create random generator
#          n = rg.rand()           # get an integer random number
#          M = rg.matrix(m,n,N)    # get random int matrix modulo N
#          M = rg.matrix(M0,N)     # with forbidden m x n matrix elements
#          C = rg.cells(m,n,s,d,N) # cell matrix of random int matrices modulo N
#          C = rg.cells(C0,M0,N)   # same but with cell/matrix args for dims
#===============================================================================

class Random:
  def __init__(self,seed):
    self.p = 1 + 2*3*5*7*11*13         # small prime number
    self.P = -1 + 2*3*5*7*11*13*17*19  # large prime number
    self.seed = seed+1

  def rand(self):
    self.seed = ((self.seed+1) * self.P) % self.p;
    return self.seed

  def matrix(self,m,n,N=None):
    if N==None:                  # called as: M = rg.matrix(M0,N)
      M0 = m;  N = n
      m = M0.size(0);
      n = M0.size(1)
    else:
      M0 = torch.zeros(m,n)-1    # no forbidden elements

    M = torch.zeros(m,n)-1       # -1 means the element is not yet defined
    for i in range(0,m):
      for j in range(0,n):
        while 1:
          r = self.rand() % N
          if M0[i,j] == r:       # forbidden matrix element
            continue
          if (M==r).any():
            continue

          M[i,j] = r
          break
    return M.int()

  def cells(self,m,n,s=None,d=None,N=None):
    if d == None:                # called as C = rg.cells(C0,M0,N)
      C0 = m;  M0 = n;  N = s
      m = C0.size(0);  n = C0.size(1)
      s = M0.size(0);  d = M0.size(1)
    else:
      M0 = torch.zeros(s,d)-1    # no forbidden elements

    C = torch.zeros(m,n,s,d)
    for i in range(0,m):
      for j in range(0,n):
         Cij = self.matrix(M0,N)
         C[i,j] = Cij
    return C.int()

#===============================================================================
# helper: peek/poke submatrix from/into matrix
# usage:  Mij = peek(M,i,j,mm,nn) peek Mij from matrix with Mij[0,0] = M[i,j]
#         C = flat(M,m,n)
#===============================================================================

def peek(M,i,j,mm,nn):  # peek sub matrix from flat matrix
  Mij = torch.zeros(mm,nn)
  for ii in range(0,mm):
    for jj in range(0,nn):
      Mij[ii,jj] = M[i+ii,j+jj]
  return Mij

def poke(M,i,j,Mij):  # peek sub matrix from flat matrix
  mm = Mij.size(0)
  nn = Mij.size(1)
  #print("poke i,j:",i,j,"mm,nn:",mm,nn)
  for ii in range(0,mm):
    for jj in range(0,nn):
      #print("M[",i+ii,",",i+jj,"] = Mij[",ii,",",jj,"]")
      M[i+ii,j+jj] = Mij[ii,jj]
  return M
