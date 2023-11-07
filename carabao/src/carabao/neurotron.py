#===============================================================================
# carabao.neurotron package: copyright: Neuronycs 2023
#===============================================================================
"""
Module carabao.neurotron supports the following classes:
    class Pulse

Example 1:
    Test a Pulse(1,2) module (1 lag cycle, 2 duty cycles) with a delta
    sequence u = [1,0,0,0,0]. The expected response is: y = [0,1,1,0,0]

pls = Pulse(1,2,log="Pulse(1,2):")
y0 = pls.feed(u0:=1)
y1 = pls.feed(u1:=0)
y2 = pls.feed(u2:=0)
y3 = pls.feed(u3:=0)
y4 = pls.feed(u4:=0)
((u0,u1,u2,u3,u4),(y0,y1,y2,y3,y4))
Pulse(1,2): 1 -> ([1,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 0 -> ([0,0],1/2) -> 1
Pulse(1,2): 0 -> ([0,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,0],0/2) -> 0
((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))

Example 2:
    Similar case but retrigger the pulse module before the niveau collapses.
    Input sequence: u = [1,0,0,0,0], desired response: y = [0,1,1,1,1]

pls = Pulse(1,2,log="Pulse(1,2):")
y0 = pls.feed(u0:=1)
y1 = pls.feed(u1:=0)
y2 = pls.feed(u2:=1)
y3 = pls.feed(u3:=0)
y4 = pls.feed(u4:=0)
((u0,u1,u2,u3,u4),(y0,y1,y2,y3,y4))
Pulse(1,2): 1 -> ([1,0],0/2) -> 0
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 1 -> ([1,0],1/2) -> 1
Pulse(1,2): 0 -> ([0,1],2/2) -> 1
Pulse(1,2): 0 -> ([0,0],1/2) -> 1
((1, 0, 1, 0, 0), (0, 1, 1, 1, 1))
"""

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse:
    """
    pulse: pulse unit
    >>> o = Pulse(lag=2,duty=3)             # mandatory args
    >>> o = Pulse(lag=2,duty=3,log="P2/3")  # set log text
    >>> y = o.feed(u:=1)                    # feed new input, receive new output
    P2/3 1 -> ([1,0,0],0/3) -> 0
    >>> u = o.inp()                         # retrieve recent input
    >>> y = o.out()                         # get pulse output
    """
    def __init__(self,lag,duty,log=None):
        def zeros(n): return [0 for k in range(0,n)]
        self.head = log                # log header
        self.n = duty                  # duty = pulse length
        self.s = zeros(lag+1)          # shift register
        self.c = 0                     # counter

    def feed(self,u):
        self.s = [u] + self.s[:-1]
        self.c = self.n if self.s[-1] > 0 else max(0,self.c-1)
        if self.head is not None: print(self)
        return self.out()

    def inp(self): return self.s[0]
    def out(self): return (self.c > 0) + 0

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self;  body = "(%s,%g/%g)" % (string(o.s),o.c,o.n)
        return o.head + " %g -> " % o.inp() + body +  " -> %g" % o.out()

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
