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
        >>> o.symbol(1)
        'B'
        >>> o.symbol([0,1,25,26,27,51,52])
        'ABZabz52'
        """
        def symb(x):
            if x < 26:
                return chr(65+x)
            elif x < 52:
                return chr(71+x)
            else:
                return str(x)

        if isinstance(x,int):
            return symb(x)
        elif isinstance(x,list):
            s = ''
            for k in range(len(x)):
                s += self.symbol(x[k])
            return s

#===============================================================================
# doctest
#===============================================================================

if __name__ == '__main__':
    import doctest            # to run doctest: $ python mod.py
    doctest.testmod()         #             or: $ python mod.py -v
