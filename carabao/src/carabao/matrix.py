import numpy as np

#===============================================================================
# class Matrix
#===============================================================================

class Matrix(np.ndarray):
    """
    class Matrix: matrix wrapper for NumPy arrays
    >>> Matrix(0,0)
    []
    >>> Matrix(2,3)
    [0 0 0; 0 0 0]
    >>> Matrix([1,2,3])
    [1 2 3]
    >>> Matrix([[1,2,3],[4,5,6]])
    [1 2 3; 4 5 6]
    """
    def __new__(cls, arg1, arg2=None, data=None):
        if isinstance(arg1,int) and arg2 is None:
            arg1 = [[arg1]]
        elif isinstance(arg1,float) and arg2 is None:
            arg1 = [[arg1]]
        elif isinstance(arg1,np.ndarray):
            #print('---- arg1',arg1,'shape/len',arg1.shape,len(arg1.shape))
            if len(arg1.shape) == 1:
                arg1 = [arg1]
        elif isinstance(arg1,int) and isinstance(arg2,int):
            arg1 = np.zeros((arg1,arg2))
        elif isinstance(arg1,list):
            if arg1 == []:
                arg1 = np.zeros((0,0))  #[[]]
            elif not isinstance(arg1[0],list):
                arg1 = [arg1]
        else:
            raise Exception('bad arg')

        #print('@@@@@ arg1/cls',arg1,cls)
        obj = np.asarray(arg1).view(cls)
        obj.custom = data
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.custom = getattr(obj, 'custom', None)

    def ______str__(self,wide=False):   # string representation of list or matrix
        return "matrix"

    def _isa(self,obj,typ=None):
        if typ is None:
            print(type(obj),type(obj).__name__)
        return (type(obj).__name__ == typ)

    def __str__(self,wide=False):   # string representation of list or matrix
        m,n = self.shape
        txt = '[';  sepi = ''
        for i in range(0,m):
            txt += sepi;  sepi = '; ';  sepj = ''
            for j in range(0,n):
                if wide == False:
                    txt += sepj + "%g" % self[i,j]
                else:
                    s = "%4g" %M[i,j].item()
                    s = s if s[0:2] != '0.' else s[1:]
                    s = s if s[0:3] != '-0.' else '-'+s[2:]
                    txt += "%5s" % s
                sepj = ' '
        txt += ']'
        return txt

    def __repr__(self):
        return self.__str__()

    def _transpose(self):
        return np.transpose(self)

    T = property(fget=_transpose)

#===============================================================================
# class Tensor
#===============================================================================

class Tensor:
    """
    class Tensor: implements a matrix of matrices (4-tensor)
    >>> T = Tensor(3,4,2,5)
    >>> T.Kmap()
    +-000/0-+-003/3-+-006/6-+-009/9-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-001/1-+-004/4-+-007/7-+-010/A-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-002/2-+-005/5-+-008/8-+-011/B-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-------+-------+-------+-------+
    >>> K = Matrix(2,5)
    >>> T = Tensor([[K,K,K,K],[K,K,K,K],[K,K,K,K]])
    >>> T.Kmap()
    +-000/0-+-003/3-+-006/6-+-009/9-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-001/1-+-004/4-+-007/7-+-010/A-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-002/2-+-005/5-+-008/8-+-011/B-+
    | 00000 | 00000 | 00000 | 00000 |
    | 00000 | 00000 | 00000 | 00000 |
    +-------+-------+-------+-------+
    """

    def __init__(self,arg=None,n=None,d=None,s=None):
        arg = 1 if arg is None else arg
        if isinstance(arg,list):
            assert len(arg) > 0
            assert isinstance(arg[0],list) and len(arg[0]) > 0
            self.data = np.array(arg)
            m = len(arg); n = len(arg[0])
            d,s = self.data[0,0].shape
        else:
            m = arg
            if n is None: n = 1
            if d is None: d = 1
            if s is None: s = 1
            lst = [[Matrix(d,s) for j in range(n)] for i in range(m)]
            self.data = np.array(lst)
        self.shape = (m,n,d,s)

    def kappa(self,i,j=None):
        """
        self.kappa():  convert matrix indices to linear index or vice versa
        >>> Tensor(4,10).kappa(i:=1,j:=3)   # k = i + j*m
        13
        >>> Tensor(4,10).kappa(k:=13)       # i = k%m, j = k//m
        (1, 3)
        """

        m,n,d,s = self.shape
        if j is None:
            k = i
            return (k%m,k//m)
        else:
            return i + j*m

    def permanence(self,p):    # encode permanence
        """
        self.permanence(p): convert permanence to symbolic string
        >>> o = Tensor(1,1)
        >>> o.permanence(0.52)
        'B'
        >>> o.permanence([-1,0,0.01,0.49,0.5,0.99,1,2])
        '<0yaAY1>'
        """
        def upper(x):
            return chr(int(65+(x-0.5)*100//2))
        def lower(x):
            return chr(int(65+32+(0.5-x)*100//2))

        if isinstance(p,list):
            s = ''
            for k in range(len(p)):
                s += self.permanence(p[k])
            return s

        if p < 0:
            return '<'
        elif p == 0:
            return '0'
        elif p == 1:
            return '1'
        elif p > 1:
            return '>'
        elif p < 0.5:
            return lower(p)
        elif p >= 0.5:
            return upper(p)
        else:
            return '?'

    def symbol(self,x):
        """
        self.symbol(x): convert index to symbol or vice versa
        >>> o = Tensor(1,1)
        >>> o.symbol(11)
        'B'
        >>> o.symbol([0,1,10,11,35,36,37,61,62])
        '01ABZabz062'
        """
        def symb(x):
            if x < 10:
                return chr(48+x)
            if x < 36:
                return chr(55+x)
            elif x < 62:
                return chr(61+x)
            else:
                return '%03g' % x

        if isinstance(x,int):
            return symb(x)
        elif isinstance(x,float):
           return symb(int(x))
        elif isinstance(x,list):
            s = ''
            for k in range(len(x)):
                s += self.symbol(x[k])
            return s

    def bar(self,n,label='',k=-1):          # bar string of length n
            if n >= 5:
                if k >= 0:
                    str = '%03g' % k
                    if len(label) > 0:
                        str += '/' + label
                else:
                    str = '---'
                while len(str) < n:
                    str += '-'
                    if len(str) < n: str = '-' + str
                return str
            if n >= 3:
                label = '-' + label
            elif n >= 5:
                label = '-' + label
            str = label
            for k in range(n-len(label)): str += '-'
            return str

    def head(self,i,n,s,width=0):
        line = '+'
        s = max(s,width)
        for j in range(n):
            if i < 0:
                sym = ''
                line += self.bar(s,'') + '+'
            else:
                k = self.kappa(i,j)
                sym = self.symbol(k)
                line += self.bar(s,sym,k) + '+'
        return line

    def _table(self,kind,I,m,n,width=0,label=''):    # print table
        """
        self.table('i',...) # for indices
        self.table('p',...) # for permanences
        self.table('w',...) # for synaptic weights
        """
        def title(n,x):
            return '-%03g-' % x

        def row(kind,I,i,j,d,s,width):
            #return '12345'
            str = ''
            for nu in range(s):
               if kind == 'i':   # index
                   str += self.symbol(I[nu])
               elif kind == 'p': # permanence
                   str += self.permanence(I[nu])
               elif kind == 'w': # permanence
                   str += '1' if I[nu] > 0.5 else '0'
               else:
                   str += '?'
            if kind == 's':
                str = I
            while len(str) < width:
                str = str + ' '
                if len(str) < width: str = ' ' + str
            return str

        #cells = self.cluster
        d = len(I[0][0])
        s = len(I[0][0][0])

        tab = ''
        for k in range(len(label)):
            tab += ' '

        str = ''
        for i in range(m):
            head = self.head(i,n,s,width)
            trailer = label if i == 0 else tab
            print(trailer+head)
            for mu in range(d):
                line = tab + '|'
                for j in range(n):
                    line += row(kind,I[i][j][mu],i,j,mu,s,width) + '|'
                print(line)
        print(tab+self.head(-1,n,s,width))

    def Pmap(self):
        m,n,d,s = self.shape
        self._table('p',self.cluster.P,m,n,width=max(s,7),label='P: ')

    def Kmap(self):
        m,n,d,s = self.shape
        self._table('i',self.data,m,n,width=max(s,7),label='')

    def Gmap(self):
        m,n,d,s = self.shape
        self._table('i',self.cluster.G,m,n,width=max(s,7),label='')

    def Wmap(self):
        m,n,d,s = self.shape
        self._table('w',self.cluster.P,m,n,width=max(s,7),label='')

#===============================================================================
# unit tests
#===============================================================================

def _case1():
    """
    >>> Matrix([])
    []
    >>> Matrix(0,0)
    []
    >>> Matrix(0,1)
    []
    >>> Matrix(0,0)
    []
    """

def _case2():
    """
    >>> Matrix([])
    []
    >>> Matrix(0,0)
    []
    >>> Matrix(0,1)
    []
    >>> Matrix(17)
    [17]
    >>> Matrix(3.14)
    [3.14]
    """

def _case3():
    """
    >>> A = Matrix([[1,2,3],[4,5,6]])
    >>> A
    [1 2 3; 4 5 6]
    >>> A._transpose()
    [1 4; 2 5; 3 6]
    >>> A.T
    [1 4; 2 5; 3 6]
    """

def _case4():
    """
    >>> T = Tensor(1)
    >>> T.Kmap()
    +-000/0-+
    |   0   |
    +-------+
    """

#===============================================================================
# udoc test
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
