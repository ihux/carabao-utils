#===============================================================================
# carabao.neurotron package: copyright: Neuronycs 2023
#===============================================================================
"""
Module carabao.pulse supports the following classes:
    class Pulse
"""

#===============================================================================
# imports
#===============================================================================

#from numpy import array, transpose
#from ypstruct import struct
#from carabao.util import repr
#from carabao.screen import Screen

#===============================================================================
# class: Pulse
#===============================================================================

class Pulse:
    """
    pulse: pulse unit with debouncer based on state machine
    >>> u=Pulse(2,3)
    >>> for i in range(6): o = u(int(i<3),'u%g:'%i)
    u0:  1 -> (1,L1,[2,3,0]) -> 0
    u1:  1 -> (2,L2,[2,3,0]) -> 0
    u2:  1 -> (2,D3,[2,3,0]) -> 1
    u3:  0 -> (1,D2,[2,3,0]) -> 1
    u4:  0 -> (0,D1,[2,3,0]) -> 1
    u5:  0 -> (0,L0,[2,3,0]) -> 0
    >>> i = u.inp()                     # retrieve recent input
    >>> o = u.out()                     # get pulse output
    >>> u.set(1)                        # set output 1 (over full duty)
    """
    def __init__(self,lag,duty,relax=0,name=None):
        self.name = name                # name header
        self.n = [lag,duty,relax]       # phase durations
        self.s = 'L'                    # state, initialized as `lag state`
        self.x = 0                      # integrator
        self.c = 0                      # counter
        self.u = 0                      # input
        self.y = 0                      # output

    def _trans(self,state):              # state transition
        l,d,r = self.n                  # get parameters
        if state == 'L':                # transition to lag state
            self.x = self.c = self.u    # init integration counter
        elif state == 'D':              # transition to duty state
            self.c = d                  # set duration counter
            raise Exception()
        elif state == 'R':              # transition to relax state
            self.c = r                  # set relax counter
        self.s = state                  # actual state change

    def integrate(self,u):
        self.u = u;
        l = self.n[0]                   # lag duration
        i = self.x + 2*u - 1            # integrator output
        self.x = max(0,min(i,l))        # limit integrator state
        return self.x,i                 # return integrator state/output

    def call(self,u):
        l,d,r = self.n                  # get parameters
        x,i = self.integrate(u)         # integrate
        y_ = int(i > l and d >0)        # proposed new output
        c_ = d if x >= l > 0 and r == 0 else self.c - 1  # proposed new count

        if self.s == 'L':               # L: lag state (debouncing)
            if y_ > 0:
                self.c = d;  self.y = y_
                self.s = 'D'
            else:
                self.c = x;  self.y = y_
        elif self.s == 'D':             # D: duty state
            if self.c <= 1 and r > 0:
                self.y = 0;  self.c = r
                self.s = 'R'            # transition to relax state
            #elif l == 0 and u > 0:
            #    self.y = y_
            #elif c_ <= 0 and r == 0:
            elif c_ <= 0 and r == 0 and u == 0:
                self.y = 0;  self.x = self.c = u
                self.s = 'L'            # transition to lag state
            else:
                #self.c = c_;
                self.c = min(c_ + u,d)
                #self.y = int(c_ > 0)
                self.y = int(self.c > 0)
        elif self.s == 'R':             # R: relax state
            if self.c <= 1 and l == 0 and u > 0:
                self.c = d;  self.y = y_
                self.s = 'D'
            elif self.c <= 1:
                self.x = self.c = u     # count down relax period
                self.s = 'L'            # transition to lag state
            else:
                self.c -= 1             # count down relax period

        if self.name is not None: print(self)
        return self.out()

    def inp(self): return self.u
    def out(self): return self.y
    def set(self,val,log=None):
        if val > 0:
            self.c = self.n[1];  self.s = 'D'
        else:
            self.trans('L')
        if log is not None:
            print(log,self)

    def __call__(self,u,log=None):
        y = self.call(u)
        if log is not None:
            print(log,self)
        return y

    def __repr__(self):
        def string(l):
            s = '['; sep = ''
            for i in range(0,len(l)): s += sep + "%g"%l[i]; sep = ','
            return s + ']'
        o = self
        body = "(%g,%s%g,%s)" % (self.x,self.s,self.c,string(self.n))
        name = self.name if self.name is not None else ""
        return name + " %g -> " % self.inp() + body +  " -> %g" % self.out()

#===============================================================================
# unit test cases:
#===============================================================================

def _case1a():
    """
    >>> P = Pulse(0,0)
    >>> for k in range(3): y = P(k<1,'')
      1 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
      0 -> (0,L0,[0,0,0]) -> 0
    """

def _case1b():                  # Identity
    """
    >>> P = Pulse(0,1)
    >>> for k in range(6): y = P(k<1 or 3 <=k <= 4,'')
      1 -> (0,D1,[0,1,0]) -> 1
      0 -> (0,L0,[0,1,0]) -> 0
      0 -> (0,L0,[0,1,0]) -> 0
      1 -> (0,D1,[0,1,0]) -> 1
      1 -> (0,D1,[0,1,0]) -> 1
      0 -> (0,L0,[0,1,0]) -> 0
    """

def _case1c():                  # Identity
    """
    >>> P = Pulse(0,1)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (0,D1,[0,1,0]) -> 1
      1 -> (0,D1,[0,1,0]) -> 1
      0 -> (0,L0,[0,1,0]) -> 0
      0 -> (0,L0,[0,1,0]) -> 0
    """

def _case1d():
    """
    >>> P = Pulse(0,1,2)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (0,D1,[0,1,2]) -> 1
      1 -> (0,R2,[0,1,2]) -> 0
      0 -> (0,R1,[0,1,2]) -> 0
      0 -> (0,L0,[0,1,2]) -> 0
    """

def _case1e():
    """
    >>> P = Pulse(0,1,2)
    >>> for k in range(4): y = P(1,'')
      1 -> (0,D1,[0,1,2]) -> 1
      1 -> (0,R2,[0,1,2]) -> 0
      1 -> (0,R1,[0,1,2]) -> 0
      1 -> (0,D1,[0,1,2]) -> 1
    """

def _case2():
    """
    >>> P = Pulse(1,0)
    >>> for k in range(3): y = P(k<1,'')
      1 -> (1,L1,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
    """

def _case3():
    """
    >>> P = Pulse(1,0)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (1,L1,[1,0,0]) -> 0
      1 -> (1,L1,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
      0 -> (0,L0,[1,0,0]) -> 0
    """

def _case4a():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(k<1,'')
      1 -> (1,L1,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
    """

def _case4b():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(k<2,'')
      1 -> (1,L1,[1,1,0]) -> 0
      1 -> (1,D1,[1,1,0]) -> 1
      0 -> (0,L0,[1,1,0]) -> 0
      0 -> (0,L0,[1,1,0]) -> 0
    """

def _case4c():
    """
    >>> P = Pulse(1,1)
    >>> for k in range(4): y = P(1,'')
      1 -> (1,L1,[1,1,0]) -> 0
      1 -> (1,D1,[1,1,0]) -> 1
      1 -> (1,D1,[1,1,0]) -> 1
      1 -> (1,D1,[1,1,0]) -> 1
    """

def _case5():
    """
    >>> P = Pulse(0,2)
    >>> for k in range(4): y = P(k<1,'')
      1 -> (0,D2,[0,2,0]) -> 1
      0 -> (0,D1,[0,2,0]) -> 1
      0 -> (0,L0,[0,2,0]) -> 0
      0 -> (0,L0,[0,2,0]) -> 0
    """

def _case6():
    """
    >>> P = Pulse(2,3)
    >>> for k in range(7): y = P(k<3,'')
      1 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      1 -> (2,D3,[2,3,0]) -> 1
      0 -> (1,D2,[2,3,0]) -> 1
      0 -> (0,D1,[2,3,0]) -> 1
      0 -> (0,L0,[2,3,0]) -> 0
      0 -> (0,L0,[2,3,0]) -> 0
    """

def _case7():
    """
    >>> P = Pulse(2,3)
    >>> for k in range(8): y = P(k<5 and k!=2,'')
      1 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      0 -> (1,L1,[2,3,0]) -> 0
      1 -> (2,L2,[2,3,0]) -> 0
      1 -> (2,D3,[2,3,0]) -> 1
      0 -> (1,D2,[2,3,0]) -> 1
      0 -> (0,D1,[2,3,0]) -> 1
      0 -> (0,L0,[2,3,0]) -> 0
    """

def _case8a():
    """
    >>> P = Pulse(0,1,3)
    >>> for k in range(8): y = P(k<4,'')
      1 -> (0,D1,[0,1,3]) -> 1
      1 -> (0,R3,[0,1,3]) -> 0
      1 -> (0,R2,[0,1,3]) -> 0
      1 -> (0,R1,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
    """

def _case8b():
    """
    >>> P = Pulse(0,1,3)
    >>> for k in range(9): y = P(k<5,'')
      1 -> (0,D1,[0,1,3]) -> 1
      1 -> (0,R3,[0,1,3]) -> 0
      1 -> (0,R2,[0,1,3]) -> 0
      1 -> (0,R1,[0,1,3]) -> 0
      1 -> (0,D1,[0,1,3]) -> 1
      0 -> (0,R3,[0,1,3]) -> 0
      0 -> (0,R2,[0,1,3]) -> 0
      0 -> (0,R1,[0,1,3]) -> 0
      0 -> (0,L0,[0,1,3]) -> 0
    """

def _case9():
    """
    >>> P = Pulse(1,1,3)
    >>> for k in range(8): y = P(k<5,'')
      1 -> (1,L1,[1,1,3]) -> 0
      1 -> (1,D1,[1,1,3]) -> 1
      1 -> (1,R3,[1,1,3]) -> 0
      1 -> (1,R2,[1,1,3]) -> 0
      1 -> (1,R1,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
      0 -> (0,L0,[1,1,3]) -> 0
    """

#===============================================================================
# doctest:
#     run: $ python neurotron.py
#     or:  $  python neurotron.py -v
#===============================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
