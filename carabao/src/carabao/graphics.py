#===============================================================================
# carabao/graphics library:
# - class Canvas
# - class Neurons
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim,ylim

#============================================================================================================
# Canvas class
# usage: can = Canvas()
#        can.circle((x,y),r,color)     # plot circle
#        can.equal()                   # set equal aspect ratio
#============================================================================================================

class Canvas:
    def __init__(self,pos):
        self.userdata = None
        self.fig = None
        self.ax = None
        self.position = pos

        self.fig,self.ax = plt.subplots()
        self.frame()
        self.ax.axis('equal')

        xy = (2,2); r = 0.5; col = 'r'
        #self.circle(xy, r, col)
        #plt.show()

    def frame(self):                        # draw frame
        xl = self.position[0];  xh = self.position[2]
        yl = self.position[1];  yh = self.position[3]
        self.ax.plot([xl,xh,xh,xl,xl],[yl,yl,yh,yh,yl],color='k',linewidth=0.5)

    def circle(self,xy,r,col=None):
        hdl = plt.Circle(xy, r, facecolor=col,edgecolor='k',linewidth=0.5)  # circle patch object
        self.ax.add_patch(hdl)               # add circle to axis' patches
        return hdl

    def equal(self):
        self.ax.axes('equal')

#============================================================================================================
# class Neurons
# usage: nr = Neurons(m,n,s,d)         # create Neurons instance
#        P = np.random.rand(s,d)       # permanences
#        nr.plot((i,j),x,y,P)
#============================================================================================================

class Neurons:
    def __init__(self,m=None,n=None,s=None,d=None):
        m = m if m != None else 4
        n = n if n != None else 10
        s = s if s != None else 5
        d = d if d != None else 2

        self.m = m;  self.n = n;  self.s = s;  self.d = d
        self.ij = (0,0)
        self.setup()

    def canvas(self):
        self.can = Canvas([0,0,self.n+1,self.m+2])
        return self.can

    def setup(self):
        self.r0 = 0.45;  self.r1 = 0.38;  self.r2 = 0.31
        self.ds = 0.11; self.rs = self.ds/3;
        self.gray = (0.8,0.8,0.8);  self.red = (1,0,0)
        self.gold = (1,0.9,0);      self.dark = (0.5,0.5,0.5)
        self.blue = (0,0.5,1);      self.green=(0,0.8,0)
        self.magenta = (1,0.2,1)

    def cell(self,ij,x=None,y=None,P=None,Q=None):
        x = x if x else 0
        y = y if y else 0
        outer = self.red if y>0 else self.gray
        inner = self.green if x>0 else self.dark

        P = P if P != None else np.random.rand(self.d,self.s)
        Q = Q if Q != None else P*0

        #print("y:",y,", outer:",outer)
        #print("x:",x,", inner:",inner)
        #print("P:\n",P)
        #print("Q:\n",Q)

        i = ij[0];  j = ij[1]
        x = 1+j; y = self.m+1-i;
        self.can.circle((x,y),self.r0,outer)
        self.can.circle((x,y),self.r1,inner)
        self.can.circle((x,y),self.r2,self.gold)

        d0 = self.d-1;  s0 = self.s-1
        for mu in range(0,self.d):
            for nu in range(0,self.s):
                xx = x + self.ds*(nu-s0/2);
                yy = y + self.ds*(d0-mu-d0/2)
                if Q[mu,nu] > 0:
                    col = self.magenta
                else:
                    col = 'w' if P[mu,nu] >= 0.5 else 'k'
                self.can.circle((xx,yy),self.rs,col)

    def input(self,j,u):
        u = u if u != None else 1
        x = 1+j; y = 1;
        col = self.blue if u > 0 else self.gray
        self.can.circle((x,y),self.r2,col)

    def at(self,i,j):  # to tell a Cell constructor where to place a cell
        self.ij = (i,j)
        return self

#============================================================================================================
# plot circle
# usage: circle(ax,(x,y),r,color)     # plot circle
#============================================================================================================
'''
def circle(ax,xy,r,col=None):
    hdl = plt.Circle(xy, 0.2, color=col)
    #fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)
    # fig = plt.gcf()
    # ax = fig.gca()

    ax.add_patch(hdl)
    return hdl
'''
#============================================================================================================
# several demos
#============================================================================================================

def demo1():
    plt.plot([1, 2, 7, 4])
    plt.ylabel('some numbers')
    plt.show()

def demo2():
    fig,ax = plt.subplots()
    ax.plot([1,2,3],[1,2,4])
    ax.axis('equal')
    circle(ax,(2,2),1,'g')

def demo3():
    can = Canvas([0,0,4,4])
    color = ('k','r','g',(0,1,1))
    for i in range(1,4):
        can.circle((i,i),0.5,color[i])
    plt.show()

def demo4():
    m = 4; n = 10; s = 5; d = 2
    r0 = 0.45;  r1 = 0.38;  r2 = 0.31
    ds = 0.11; rs = ds/3;
    gray = (0.8,0.8,0.8);  red = (1,0,0)
    gold = (1,0.9,0);      dark = (0.5,0.5,0.5)
    blue = (0,0.5,1);      green=(0,0.8,0)

    can = Canvas([0,0,11,m+2])

    for j in range(0,n):
        x = 1+j; y = 1;
        #plt.plot([x,x],[1,m+1],color=blue,linewidth=3)
        can.circle((x,y),r2,blue)

    for i in range(0,m):
        for j in range(0,n):
            x = 1+j; y = m+1-i;
            can.circle((x,y),r0,red)
            can.circle((x,y),r1,green)
            can.circle((x,y),r2,gold)

            for mu in range(0,d):
                for nu in range(0,s):
                    xx = x + ds*(nu-(s-1)/2);  yy = y + ds*(mu-(d-1)/2)
                    p = np.random.rand()       # permanence
                    if p < 0.5:
                        can.circle((xx,yy),rs,'k')
                    else:
                        can.circle((xx,yy),rs,'w')
    plt.show()


def demo5():
    nrs = Neurons(2,2,5,2)
    can = nrs.canvas()                    # create/open related canvas
    nrs.cell((0,0),0,0)
    nrs.cell((0,1),0,1)
    nrs.cell((1,0),1,0)
    nrs.cell((1,1),1,1)
    nrs.input(0)
    nrs.input(1)
    plt.show()
