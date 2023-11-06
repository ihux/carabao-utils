#===============================================================================
# carabao/neurotron package: copyright: Neuronycs 2023
# - class Pulse
#===============================================================================

#=======================================================================================
# class: Pulse
#=======================================================================================

class Pulse:
    """
    pulse: pulse unit
    >>> o = Pulse(lag=2,duty=3)                # mandatory args
    >>> o = Pulse(lag=2,duty=3,log="P2/3")     # set log text
    >>> y = o.feed(u)                          # feed with new input, return new output
    >>> u = o.inp()                            # retrieve recent input
    >>> y = o.out()                            # get pulse output
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
