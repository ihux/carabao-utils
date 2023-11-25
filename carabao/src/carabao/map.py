#===============================================================================
# class Map
#===============================================================================

class Map:
    def __init__(self,cluster=None):
        self.cluster = cluster
        self.shape = cluster.shape if cluster is not None else (4,10,2,5)

    def zeros(self,d,s):
        return [[0 for j in range(s)] for i in range(d)]

    def kappa(self,i,j=None):
        """
        self.kappa():  convert matrix indices to linear index or vice versa
        >>> Map().kappa(i:=1,j:=3)   # k = i+j*m
        13
        >>> Map().kappa(k:=13)       # i = k%m, j = k//m
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
        >>> o = Map()
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
        >>> o = Map()
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

    def table(self,kind,I,m,n,width=0,label=''):    # print table
        """
        self.table('i',...) # for indices
        self.table('p',...) # for permanences
        self.table('w',...) # for synaptic weights
        """
        def title(n,x):
            return '-%03g-' % x

        def row(kind,I,i,j,d,s,width):
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

        cells = self.cluster
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
        m,n,d,s = cells.shape
        self.table('p',self.cluster.P,m,n,width=max(s,7),label='P: ')

    def Kmap(self):
        m,n,d,s = cells.shape
        self.table('i',self.cluster.K,m,n,width=max(s,7),label='K: ')

    def Gmap(self):
        m,n,d,s = cells.shape
        self.table('i',self.cluster.G,m,n,width=max(s,7),label='G: ')

    def Wmap(self):
        m,n,d,s = cells.shape
        self.table('w',self.cluster.P,m,n,width=max(s,7),label='W: ')

    def Fmap(self):
        m,n,d,s = cells.shape
        self.table('w',self.cluster.F,m,n,width=max(s,7),label='F: ')

    def Smap(self):             # state map
        m,n,d,s = self.shape
        S = self.zeros(m,n)
        cells = self.cluster
        for i in range(m):
            for j in range(n):
                states = ['-','-','-','-','-']
                if cells.U[i,j]: states[0] = 'U'
                if cells.Q[i,j]: states[0] = 'Q'
                if cells.X[i,j]: states[1] = 'X'
                if cells.S[i,j]: states[2] = 'S'
                if cells.L[i,j]: states[2] = 'L'
                if cells.D[i,j]: states[3] = 'D'
                if cells.B[i,j]: states[3] = 'B'
                if cells.Y[i,j]: states[4] = 'Y'
                str = ''
                for k in range(len(states)):
                    str += states[k]
                S[i][j] = [str]
        self.table('s',S,m,n,width=max(s,7),label='S: ')
