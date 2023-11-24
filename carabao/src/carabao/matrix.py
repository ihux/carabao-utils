"""
module matrix:
- class Attribute      # easy get/set of object attributes
- class Matrix         # MATLAB style matrix class (NumPy based)
- class Field          # matrix of matrices

Attribute methods:
- get
- set

Matrix methods:
- construct             # Matrix construction
- kappa                 # conversion between linear <-> quadratic index
- transpose             # matrix transposition
- indexing              # by scalar index or index pair
- slicing               # indexing by slices
- column                # convert matrix to column
- reshape               # reshape matrix
- string representation # represent matrix as a string
- mul                   # element wise matrix multiplication
- matmul                # algebraic matrix multiplication

Field methods:
- construct             # Field construction
- kappa                 # conversion between linear <-> quadratic index
- permanence            # cnvert permanence to symbolic string
- symbol                # convert symbol to index or vice versa
- bar                   # create a labeled bar
- head                  # create a cell head
- imap                  # create index map
- vmap                  # create value map

Utility functions:
- eye                   # unit matrix
- zeros                 # zero matrix
- one                   # one matrix
- rand                  # random matrix
- seed                  # set random seed
- max                   # row of column maxima
- min                   # row of column minima
- size                  # matrix sizes
- magic                 # magic matrix
- sum                   # row of column sum
- any                   # row of column any's
- all                   # row of column all's
- length                # maximum size
- isempty               # check if matrix is empty

"""

import numpy as np

#===============================================================================
# class Attribute
#===============================================================================

class Attribute:
    def get(self,tags):
        """
        >>> o = Attribute()
        >>> o.set('A,B,C',(1,2,3))
        >>> o.get('A')
        1
        >>> o.get('A,B,C')
        (1, 2, 3)
        """
        out = ()
        while True:
            idx = tags.find(',')
            if idx < 0:
                tag = tags;  tags = ''
            else:
                tag = tags[:idx]
                tags = tags[idx+1:]
            arg = getattr(self,tag,None)
            out = out + (arg,)
            if tags == '':
                return out if len(out) > 1 else out[0]

    def set(self,tags,args):
        """
        >>> o = Attribute()
        >>> o.set('A,B,C',(7,8,9))
        >>> o.get('A,B,C')
        (7, 8, 9)
        >>> o.set('X',((5,2),))
        >>> o.get('X')
        (5, 2)
        """
        if not isinstance(args,tuple):
            args = (args,)
        for k in range(len(args)):
            idx = tags.find(',')
            if idx < 0:
                tag = tags;  tags = ''
            else:
                tag = tags[:idx]
                tags = tags[idx+1:]
            setattr(self,tag,args[k])
            if tags == '':
                if len(args) > k+1:
                    raise Exception('too many values provided by arg2 tuple')
                return


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
    >>> Matrix(range(5))
    [0 1 2 3 4]

    See also: Matrix, eye, zeros, ones
    """
    def __new__(cls, arg1=None, arg2=None, data=None):
        isa = isinstance
        arg1 = [] if arg1 is None else arg1
        if isa(arg1,int) and arg2 is None:
            arg1 = [[arg1]]
        elif isa(arg1,float) and arg2 is None:
            arg1 = [[arg1]]
        elif isa(arg1,np.ndarray):
            if len(arg1.shape) == 1:
                arg1 = [arg1]
        elif isa(arg1,int) and isa(arg2,int):
            arg1 = np.zeros((arg1,arg2))
        elif isa(arg1,list):
            if arg1 == []:
                arg1 = np.zeros((0,0))  #[[]]
            elif not isa(arg1[0],list):
                arg1 = [arg1]
        elif isa(arg1,range):
            arg1 = np.array([arg1])
            #print('### arg1:',arg1)
            return Matrix(arg1)
        else:
            raise Exception('bad arg')

        obj = np.asarray(arg1).view(cls)
        obj.custom = data
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.custom = getattr(obj, 'custom', None)

    def kappa(self,i,j=None):
        """
        Matrix.kappa():  convert matrix indices to linear index or vice versa
        >>> Matrix(4,10).kappa(i:=1,j:=3)   # k = i + j*m
        13
        >>> Matrix(4,10).kappa(k:=13)       # i = k%m, j = k//m
        (1, 3)
        """
        m,n = self.shape
        if j is None:
            k = i
            return (k%m,k//m)
        else:
            return i + j*m

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

#    def __max__(self):
#       m,n = self.shape
#        return 0

    def __getitem__(self,idx):
        isa = isinstance  # shorthand
        if isa(idx,int):
            i,j = self.kappa(idx)
            result = super().__getitem__((i,j))
            iresult = int(result)
            if result == iresult: return iresult
            return result
        elif isa(idx,tuple):
            i,j = idx;
            m,n = self.shape
            if isa(i,int) and isa(j,slice):
                if i < 0 or i >= m:
                    raise Exception('row index out of range')
                idx = (slice(i,i+1,None),j)
            elif isa(i,slice) and isa(j,int):
                if j < 0 or j >= n:
                    raise Exception('column index out of range')
                idx = (i,slice(j,j+1,None))
        elif isa(idx,Matrix):
            #print('Matrix:',idx)
            m,n = idx.shape
            result = Matrix(m,n)
            for i in range(m):
                for j in range(n):
                    k = idx[i,j]
                    mu,nu = self.kappa(k)
                    result[i,j] = super().__getitem__((mu,nu))
                    #print('result[%g,%g]'%(i,j), 'k:',k,'(mu,nu)',(mu,nu))
            return result
        result = super().__getitem__(idx)
        if isa(result,np.int64):
            iresult = int(result)
            if result == iresult: return iresult
        return result

    def __call__(self): # convert to column vector
        """
        A = magic(2)
        A()
        [1; 3; 4; 2]
        """
        m,n = self.shape
        out = Matrix(m*n,1)
        for i in range(m):
            for j in range(n):
                k = self.kappa(i,j)
                out[k,0] = super().__getitem__((i,j))
        return out

    def __mul__(self,other):
        """
        >>> A = magic(2); B = A.T; print(A)
        [1 3; 4 2]
        >>> A*B
        [1 12; 12 4]
        >>> A*5
        [5 15; 20 10]
        >>> 3*A
        [3 9; 12 6]
        """
        isa = isinstance
        if isa(other,Matrix):
            if self.shape != other.shape:
                raise Exception('Matrix.__mul__: incompatible sizes')
        return super().__mul__(other)

    def reshape(self,m,n): # convert to column vector
        """
        >>> A = magic(4)[:3,:]; print(A)
        [16 2 3 13; 5 11 10 8; 9 7 6 12]
        >>> B = A(); print(B)
        [16; 5; 9; 2; 11; 7; 3; 10; 6; 13; 8; 12]
        >>> B[2]
        9
        >>> B.reshape(3,4)
        [16 2 3 13; 5 11 10 8; 9 7 6 12]
        >>> B.reshape(6,2)
        [16 3; 5 10; 9 6; 2 13; 11 8; 7 12]
        >>> B.reshape(2,6)
        [16 9 11 3 6 8; 5 2 7 10 13 12]
        >>> B.reshape(1,12)
        [16 5 9 2 11 7 3 10 6 13 8 12]
        """
        v = self()  # convert to column
        #print('### shape:',v.shape,'v:',v)
        mn = v.shape[0]
        if mn != m*n:
            raise Exception('incompatible dimensions for reshape')
        out = Matrix(m,n)
        for k in range(mn):
            i,j = out.kappa(k)
            #print('### i,j:',i,j)
            out[i,j] = v[k,0]
        return out

    T = property(fget=_transpose)


#===============================================================================
# class Field
#===============================================================================

class Field:
    """
    class Field: implements a matrix of matrices (4-tensor)
    >>> T = Field(3,4,2,5)
    >>> T.imap()
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
    >>> T = Field([[K,K,K,K],[K,K,K,K],[K,K,K,K]])
    >>> T.imap()
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
        >>> Field(4,10).kappa(i:=1,j:=3)   # k = i + j*m
        13
        >>> Field(4,10).kappa(k:=13)       # i = k%m, j = k//m
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
        >>> o = Field(1,1)
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
        >>> o = Field(1,1)
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
        s = _max(s,width)
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

    def vmap(self):
        m,n,d,s = self.shape
        self._table('p',self.data,m,n,width=_max(s,7),label='P: ')

    def imap(self):
        m,n,d,s = self.shape
        self._table('i',self.data,m,n,width=_max(s,7),label='')

    def _Gmap(self):
        m,n,d,s = self.shape
        self._table('i',self.cluster.G,m,n,width=_max(s,7),label='')

    def _Wmap(self):
        m,n,d,s = self.shape
        self._table('w',self.cluster.P,m,n,width=_max(s,7),label='')

#===============================================================================
# helper functions
#===============================================================================

def _max(x,y):
    return x if x > y else y

def _min(x,y):
    return x if x < y else y

#===============================================================================
# matrix functions
#===============================================================================

def eye(n):
    """
    >>> eye(3)
    [1 0 0; 0 1 0; 0 0 1]
    """
    I = Matrix(n,n)
    for k in range(n):
        I[k,k] = 1
    return I

def zeros(m,n=None):
    """
    >>> zeros(3)
    [0 0 0; 0 0 0; 0 0 0]
    >>> zeros(2,4)
    [0 0 0 0; 0 0 0 0]
    """
    n = m if n is None else n
    if m == 0 or n == 0:
        return Matrix(m,n)
    return Matrix(m,n)

def ones(m,n=None):
    """
    >>> ones(3)
    [1 1 1; 1 1 1; 1 1 1]
    >>> ones(2,3)
    [1 1 1; 1 1 1]
    >>> ones(0,0)
    []
    """
    n = m if n is None else n
    if m == 0 or n == 0:
        return Matrix(m,n)
    return Matrix(m,n) + 1

def rand(m,n=None,modulus=None):
    """
    >>> seed(0)
    >>> rand(2)
    [0.548814 0.715189; 0.602763 0.544883]
    >>> rand(2,3)
    [0.423655 0.645894 0.437587; 0.891773 0.963663 0.383442]
    >>> rand(0,0)
    []
    """
    n = m if n is None else n
    if m == 0 or n == 0:
        return Matrix(m,n)
    R = Matrix(m,n)

    if modulus is None:
        for i in range(m):
            for j in range(n):
                R[i,j] = np.random.rand()
    else:
        modulus = int(modulus)
        for i in range(m):
            for j in range(n):
                R[i,j] = np.random.randint(modulus)
    return R

def seed(s):
    """
    seed(): set random seed
    >>> seed(0)
    """
    np.random.seed(s)

def size(arg):
    """
    size(): matrix sizes
    >>> size(3.14)
    (1, 1)
    >>> size(Matrix(3,5))
    (3, 5)
    >>> size([])
    (0, 0)
    """
    if isinstance(arg,list) and len(arg) == 0:
        return (0,0)
    elif isinstance(arg,int) or isinstance(arg,float):
        return (1,1)
    elif isinstance(arg,Matrix):
        m,n = arg.shape
        return (m,n)
    else:
        raise Exception('bad type')

def max(arg1,arg2=None):
    """
    >>> max(2,3)
    3
    >>> max(2,3.6)
    3.6
    >>> A = Matrix([[1,3,-2],[0,2,-1]])
    >>> B = Matrix([[8,2,-3],[1,0,-2]])
    >>> max(A,B)
    [8 3 -2; 1 2 -1]
    >>> max(A)
    [1 3 -1]
    >>> max(2,A)
    [2 3 2; 2 2 2]
    >>> max(B,1)
    [8 2 1; 1 1 1]
    """
    scalar1 = isinstance(arg1,int) or isinstance(arg1,float)
    scalar2 = isinstance(arg2,int) or isinstance(arg2,float)

    if scalar1:
        if arg2 is None:
            return arg1
        elif scalar2:
            return _max(arg1,arg2)
        else:
            arg1 = arg1 + 0*arg2
    elif scalar2:
        arg2 = arg2 + 0*arg1

    m,n = arg1.shape
    if arg2 is None:
        M = Matrix(1,n)
        for j in range(n):
            maxi = arg1[0,j]
            for i in range(1,m):
                maxi = _max(arg1[i,j],maxi)
            M[0,j] = maxi
        return M
    else:
        assert arg1.shape == arg2.shape
        M = Matrix(m,n)
        for i in range(m):
            for j in range(n):
                M[i,j] = _max(arg1[i,j],arg2[i,j])
        return M.item() if m == 1 and n == 1 else M

def min(arg1,arg2=None):
    """
    >>> min(2,3)
    2
    >>> min(2.1,3)
    2.1
    >>> A = Matrix([[1,3,-2],[0,2,-1]])
    >>> B = Matrix([[8,2,-3],[1,0,-2]])
    >>> min(A,B)
    [1 2 -3; 0 0 -2]
    >>> min(A)
    [0 2 -2]
    >>> min(2,B)
    [2 2 -3; 1 0 -2]
    >>> min(A,1)
    [1 1 -2; 0 1 -1]
    """
    scalar1 = isinstance(arg1,int) or isinstance(arg1,float)
    scalar2 = isinstance(arg2,int) or isinstance(arg2,float)

    if scalar1:
        if arg2 is None:
            return arg1
        elif scalar2:
            return _min(arg1,arg2)
        else:
            arg1 = arg1 + 0*arg2
    elif scalar2:
        arg2 = arg2 + 0*arg1

    m,n = arg1.shape
    if arg2 is None:
        M = Matrix(1,n)
        for j in range(n):
            maxi = arg1[0,j]
            for i in range(1,m):
                maxi = _min(arg1[i,j],maxi)
            M[0,j] = maxi
        return M
    else:
        assert arg1.shape == arg2.shape
        M = Matrix(m,n)
        for i in range(m):
            for j in range(n):
                M[i,j] = _min(arg1[i,j],arg2[i,j])
        return M.item() if m == 1 and n == 1 else M

def magic(n):
    """
    >>> magic(0)
    []
    >>> magic(1)
    1
    >>> magic(2)
    [1 3; 4 2]
    >>> magic(3)
    [8 1 6; 3 5 7; 4 9 2]
    >>> magic(4)
    [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]
    """
    if n == 0:
        return []
    elif n == 1:
        return 1
    elif n == 2:
        return Matrix([[1,3],[4,2]])
    elif n == 3:
        return Matrix([[8,1,6],[3,5,7],[4,9,2]])
    elif n == 4:
        return Matrix([[16,2,3,13],[5,11,10,8],[9,7,6,12],[4,14,15,1]])
    else:
        raise Exception('n > 4 not supported')

def sum(arg):
    """
    >>> sum(2)
    2
    >>> sum(3.14)
    3.14
    >>> sum([1,2,3])
    6
    >>> A = magic(4)[:3,:];  print(A)
    [16 2 3 13; 5 11 10 8; 9 7 6 12]
    >>> sum(A)
    [30 20 19 33]
    >>> sum(A[:,1])
    20
    >>> sum(A[2,:])
    34
    >>> sum(True)
    1
    >>> C=ones(1,4)
    >>> sum(C>0)
    4
    """
    isa = isinstance
    if isa(arg,int) or isa(arg,np.int64) or isa(arg,float):
        return arg
    elif isa(arg,list):
        M = Matrix(arg)
        return sum(M)
    elif isa(arg,Matrix):
        #print('##### Matrix:',arg)
        m,n = arg.shape
        if m == 0 or n == 0:
            return []
        elif m == 1:
            s = 0
            #print('##### row:',arg,'s:',s)
            for j in range(n): s += arg[0,j]
            return s
        elif n == 1:
            s = 0
            for i in range(m): s += arg[i,0]
            return s
        else:
            out = Matrix(1,n)
            for j in range(n):
                s = 0
                for i in range(m):
                    s += arg[i,j]
                out[0,j] = s
            return out
    else:
        return arg.sum()


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
    >>> Matrix()
    []
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
    >>> T = Field(1)
    >>> T.imap()
    +-000/0-+
    |   0   |
    +-------+
    """

def _case4():
    """
    >>> A = Matrix([[1,3,-2],[0,2,-1]])
    >>> max(0,min(A,1))
    [1 1 0; 0 1 0]
    """

def _case5a():  # indexing with slices
    """
    >>> A = magic(4); print(A)
    [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]
    >>> A[0,0]
    16
    >>> B = A[:3,:]; print(B)
    [16 2 3 13; 5 11 10 8; 9 7 6 12]
    >>> B[0,:]
    [16 2 3 13]
    >>> B[1,:]
    [5 11 10 8]
    >>> B[2,:]
    [9 7 6 12]
    >>> B[:,0]
    [16; 5; 9]
    >>> B[:,1]
    [2; 11; 7]
    >>> B[:,2]
    [3; 10; 6]
    >>> B[:,3]
    [13; 8; 12]
    """

def _case5b():  # indexing with slices, column ranges
    """
    >>> A = magic(4); print(A)
    [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]
    >>> B = A[:3,:]; print(B)
    [16 2 3 13; 5 11 10 8; 9 7 6 12]
    >>> B[:,:]
    [16 2 3 13; 5 11 10 8; 9 7 6 12]
    >>> B[:,:2]
    [16 2; 5 11; 9 7]
    >>> B[:,1:4:2]
    [2 13; 11 8; 7 12]
    """
def _case5c():  # indexing with slices, row ranges
    """
    >>> A = magic(4); print(A)
    [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]
    >>> C = A[:3,:].T; print(C)
    [16 5 9; 2 11 7; 3 10 6; 13 8 12]
    >>> C[:,:]
    [16 5 9; 2 11 7; 3 10 6; 13 8 12]
    >>> C[:2,:]
    [16 5 9; 2 11 7]
    >>> C[1:4:2,:]
    [2 11 7; 13 8 12]
    """
def _case5d():  # indexing with slices, row & column ranges
    """
    >>> A = magic(4); print(A)
    [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]
    >>> A[:2,:2]
    [16 2; 5 11]
    >>> A[1:4:2,1:3]
    [11 10; 14 15]
    >>> A[1:3,1:4:2]
    [11 8; 7 12]
    """

#===============================================================================
# udoc test
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
