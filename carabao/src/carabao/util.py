#===============================================================================
# carabao/util.py: carabao utilities
#    Random()    random generator class
#    peek()      peek submatrix from flat matrix
#    poke()      poke submatrix to flat matrix
#    squeeze()   squeeze flat matrix into mxn cell matrix
#    isa()       check object for specific type name
#    column()    create mx1 column matrix from list
#    sat()       truncates every matrix element to range [0,1]
#    repr()      representation string of numpy array
#===============================================================================

from numpy import transpose, array, zeros

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
  """
  class Random: portable pseudo random generator for random integers modulo N
  >>> rg = Random(seed=88)             # create random generator
  >>> rg.rand()                        # get integer random number
  871
  >>> rg.matrix(m=2,n=3,N=6)           # get random int matrix modulo N
  array([[4., 0., 2.],
         [5., 1., 3.]])
  """
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
      M0 = zeros((m,n))-1    # no forbidden elements

    M = zeros((m,n))-1       # -1 means the element is not yet defined
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
    return M

  def cells(self,m,n,s=None,d=None,N=None):
    if d == None:                # called as C = rg.cells(C0,M0,N)
      C0 = m;  M0 = n;  N = s
      m = C0.size(0);  n = C0.size(1)
      s = M0.size(0);  d = M0.size(1)
    else:
      M0 = zeros(s,d)-1    # no forbidden elements

    C = zeros(m,n,s,d)
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
    Mij = zeros(mm,nn)
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

#===============================================================================
# flatten a cell matrix (matrix of nxn matrices), or squeeze a flat matrix
# - usage: M = flat(C)          # flatten cell matrix C
#          C = squeeze(M,m,n)   # squeeze a flat matrix into a cell matrix
#===============================================================================

def flat(C):  # M = flat(C)
    mC = C.size(0);  nC = C.size(1)
    if mC*nC == 0:
      M = zeros(0,0)
      return
    M00 = C[0,0]
    m = M00.size(0);  n = M00.size(1)
    M = zeros(mC*m,nC*n)
    for i in range(0,mC):
      for j in range(0,nC):
        Mij = C[i,j]
        M = poke(M,i,j,Mij)
    return M

#===============================================================================
# squeeze a flat matrix M into mxn cell matrix
# - usage: C = squeeze(M,m,n)   # squeeze a flat matrix into a mxn cell matrix
#===============================================================================

def squeeze(M, m, n):     # C = squeeze(M,m,n)
    mm = int(M.size(0)/m)
    nn = int(M.size(1)/n)
    C = zeros(m,n,mm,nn)-1
    for i in range(0,m):
      for j in range(0,n):
        Mij = sw.peek(M,i,j,mm,nn)
        C[i,j] = Mij
    return C

#===============================================================================
# utility: check object for specific type name
#===============================================================================

def isa(obj,typ=None):
    """
    isa(): has object certain type?

        ok = isa("junk",'string')  # => True
        ok = isa([1,2,3],'list')   # => True
        ok = isa((2,3),'tuple')    # => True
    """
    if typ is None:
        print(type(obj),type(obj).__name__)
    return (type(obj).__name__ == typ)

#===============================================================================
# helper: create column vector from list
#===============================================================================

def column(x):
    """
    column(): create column vector from list

        v = column([0,1])
    """
    return transpose(array([x]))

#===============================================================================
# helper: sat function for a numpy matrix
# - truncates every matrix element to range 0.0 ... 1.0
#===============================================================================

def sat(X):
    def lt1(X): return 1 + (X-1<=0)*(X-1)
    def gt0(X): return (X>=0)*X
    return lt1(gt0(X))

#===============================================================================
# utility: check object for specific type name
#===============================================================================

def repr(obj,wide=False):   # string representation of list or matrix
    if isa(obj,'list'):
        txt = "[";  M = array([obj])
        #print("repr M:",M,type(M),M.shape)
    elif isa(obj,'ndarray'):
        txt = "#[";  M = obj
        if len(M.shape) == 1:
            txt = ":[";  M = obj
            M = array([M])
    else:
        return obj

    m,n = M.shape
    sepi = ''
    for i in range(0,m):
        txt += sepi;  sepi = '; ';  sepj = ''
        for j in range(0,n):
            if wide == False:
                txt += sepj + "%g" % M[i,j]
            else:
                s = "%4g" %M[i,j].item()
                s = s if s[0:2] != '0.' else s[1:]
                s = s if s[0:3] != '-0.' else '-'+s[2:]
                txt += "%5s" % s
            sepj = ' '
    txt += ']'
    return txt

#===============================================================================
# utility: check if argument is scalar (either in or float)
#===============================================================================

def isscalar(x):
    """
    isscalar(): check if arg is either int or float
    >>> isscalar(5)
    True
    >>> isscalar(3.14)
    True
    >>> import numpy
    >>> isscalar(numpy.array([1.2]))
    False
    """
    return isinstance(x,int) or isinstance(x,float)

#===============================================================================
# doctest
#===============================================================================

if __name__ == '__main__':
    import doctest            # to run doctest: $ python mod.py
    doctest.testmod()         #             or: $ python mod.py -v
