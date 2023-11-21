"""
module cluster:
    class Map        mapping of data values
    class Cluster    managing parameters of network cluster
"""

#===============================================================================
# class Map
#===============================================================================

class Map:
    def __init__(self,cluster=None):
        self.cluster = cluster

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
        '01ABZabz62'
        """
        def symb(x):
            if x < 10:
                return chr(48+x)
            if x < 36:
                return chr(55+x)
            elif x < 62:
                return chr(61+x)
            else:
                return str(x)

        if isinstance(x,int):
            return symb(x)
        elif isinstance(x,list):
            s = ''
            for k in range(len(x)):
                s += self.symbol(x[k])
            return s

    def bar(self,n):                   # bar string of length n
            str = ''
            for k in range(n): str += '-'
            return str

    def head(self,n,c):                # draw a box head with label
        str = ''
        for k in range(n):
            if k == 1:
                str += '('
            elif k == 2:
                str += c
            elif k == 3:
                str += ')'
            else:
                str += '-'
        return str

    def headline(self,n,s):
        line = '+'
        for j in range(n):
            #line += title(s,23) + '+'
            line += self.bar(s) + '+'
        return line

    def Pmap(self):

        def title(n,x):
            return '-%03g-' % x

        def weights(cells,i,j,d):
            m,n,dd,s = cells.shape
            W = cells.P[i][j][d]
            str = ''
            for k in range(s):
               str += self.permanence(W[k])
            return str

        cells = self.cluster
        m,n,d,s = cells.shape
        head = self.headline(n,s)
        str = ''
        for i in range(m):
            print(head)
            for mu in range(d):
                line = '|'
                for j in range(n):
                    line += weights(cells,i,j,mu) + '|'
                print(line)
        print(head)

    def Kmap(self):

        def title(n,x):
            return '-%03g-' % x

        def indices(cells,i,j,d):
            m,n,dd,s = cells.shape
            K = cells.K[i][j][d]
            str = ''
            for k in range(s):
               str += self.symbol(K[k])
            return str

        cells = self.cluster
        m,n,d,s = cells.shape
        head = self.headline(n,s)
        str = ''
        for i in range(m):
            print(head)
            for mu in range(d):
                line = '|'
                for j in range(n):
                    line += indices(cells,i,j,mu) + '|'
                print(line)
        print(head)

#===============================================================================
# doctest
#===============================================================================

if __name__ == '__main__':
    import doctest            # to run doctest: $ python mod.py
    doctest.testmod()         #             or: $ python mod.py -v
