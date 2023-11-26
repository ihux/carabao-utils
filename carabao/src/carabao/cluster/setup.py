"""
module carabao.cluster.setup
    class Collab  # setup collaboration terminal
"""

from carabao.matrix import Matrix,Field,ones

#===============================================================================
# class Collab
#===============================================================================

class Collab:  # to manage collaboration topology
    """
    >>> shape = (3,4,2,5)
    >>> Collab(*shape)
    Collab(3,4)
    >>> Collab(*shape).map()
    K: +-000/0-+-003/3-+-006/6-+-009/9-+
       |  12   |  45   |  78   |  AB   |
       +-001/1-+-004/4-+-007/7-+-010/A-+
       |  02   |  35   |  68   |  9B   |
       +-002/2-+-005/5-+-008/8-+-011/B-+
       |  01   |  34   |  67   |  9A   |
       +-------+-------+-------+-------+
    P: +-000/0-+-003/3-+-006/6-+-009/9-+
       |  11   |  11   |  11   |  11   |
       +-001/1-+-004/4-+-007/7-+-010/A-+
       |  11   |  11   |  11   |  11   |
       +-002/2-+-005/5-+-008/8-+-011/B-+
       |  11   |  11   |  11   |  11   |
       +-------+-------+-------+-------+
    W: +-000/0-+-003/3-+-006/6-+-009/9-+
       |  11   |  11   |  11   |  11   |
       +-001/1-+-004/4-+-007/7-+-010/A-+
       |  11   |  11   |  11   |  11   |
       +-002/2-+-005/5-+-008/8-+-011/B-+
       |  11   |  11   |  11   |  11   |
       +-------+-------+-------+-------+
    """
    def __init__(self,m,n,dummy1=0,dummy2=0):
        self.shape = (m,n)
        self.init()

    def __str__(self):
        return 'Collab(%g,%g)' % self.shape

    def __repr__(self):
        return self.__str__()

    def _Kij(self,i,j):
        m,n = self.shape
        Kij = Matrix(1,m-1)
        s = 0
        for l in range(m):
            if l != i:
                Kij[s] = l + m*j;  s += 1
        return Kij

    def init(self):
        m,n = self.shape
        self.K = Field(m,n,1,m-1)
        self.P = Field(m,n,1,m-1)
        self.W = Field(m,n,1,m-1)
        for i in range(m):
            for j in range(n):
                self.K[i,j] = self._Kij(i,j)
                self.P[i,j] = ones(1,m-1)
                self.W[i,j] = ones(1,m-1)

    def map(self):
        self.K.imap('K: ')
        self.P.vmap('P: ')
        self.W.imap('W: ')

#===============================================================================
# doc test
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
