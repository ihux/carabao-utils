import numpy as np

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

    def t(self):                       # transpose
        return np.transpose(self)

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
    >>> A.t()
    [1 4; 2 5; 3 6]
    """

#===============================================================================
# udoc test
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
