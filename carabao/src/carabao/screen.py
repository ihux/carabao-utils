#===============================================================================
# carabao/screen library:
# - class Screen (copyright: Neuronycs 2023)
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.transforms import Affine2D

#===============================================================================
# Canvas class
# usage: can = Canvas()
#        can.circle((x,y),r,color)     # plot circle
#        can.equal()                   # set equal aspect ratio
#===============================================================================

class Canvas:
    def __init__(self,pos=None):
        pos = [0,0,1,1] if pos == None else pos
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
        hdl = plt.Circle(xy, r, facecolor=col,edgecolor='k',linewidth=0.5)
        self.ax.add_patch(hdl)               # add circle to axis' patches
        return hdl

    def rect(self,xy1,xy2,col=None,angle=None):
        angle = 0 if angle == None else angle
        width = xy2[0]-xy1[0]; height = xy2[1]-xy1[1];
        hdl = plt.Rectangle(xy1, width,height,
                            facecolor=col, edgecolor=col,angle=angle)
        self.ax.add_patch(hdl)               # add rectangle to axis' patches
        return hdl

    def fancy(self,xy1,xy2,col=None,r=None,angle=None,center=None):
        r = 0.1 if r == None else r
        angle = 0 if angle == None else angle
        center = (0,0) if center == None else center
        lw = 0.5
        fcol = col
        ecol = 'k'
        style = "round, pad=%g" % r

        center = (center[0]+(xy1[0]+xy2[0])/2, center[1]+(xy1[1]+xy2[1])/2)
        ts = self.ax.transData
        tr = Affine2D().rotate_deg_around(center[0],center[1],angle)
        t = tr + ts

        xy = (xy1[0]+r,xy1[1]+r)
        hdl = patches.FancyBboxPatch(xy, xy2[0]-xy1[0]-2*r,xy2[1]-xy1[1]-2*r,
                    facecolor=fcol, edgecolor=ecol, linewidth=lw,
                    boxstyle=style, transform=t)

        trans = Affine2D().rotate_deg_around(center[0],center[1],angle) + self.ax.transData
        hdl.set_transform(trans)

        self.ax.add_patch(hdl)               # add rectangle to axis' patches
        return hdl

    def poly(self,points,facecolor=None,edgecolor=None,linewidth=None):
        facecolor = (0.5,0.5,0.5) if facecolor == None else facecolor
        edgecolor = 'k' if edgecolor == None else edgecolor
        linewidth = 0.5 if linewidth == None else linewidth

        hdl = patches.Polygon(points, facecolor=facecolor, edgecolor=edgecolor,
                              linewidth=linewidth)
        self.ax.add_patch(hdl)               # add rectangle to axis' patches
        return hdl

    def equal(self):
        self.ax.axes('equal')

    def show(self):
        plt.show()


#===============================================================================
# class Screen
# usage: scr = Screen(m,n,s,d)         # create Screen instance
#        P = np.random.rand(s,d)       # permanences
#        Q = (P >= 0.5)                # synaptics
#        scr.plot((i,j),x,y,P,Q)
#===============================================================================

class Screen:
    def __init__(self,tag,m=None,n=None,s=None,d=None):
        m = m if m != None else 4
        n = n if n != None else 10
        s = s if s != None else 5
        d = d if d != None else 2

        self.tag = tag
        self.m = m;  self.n = n;  self.s = s;  self.d = d
        self.ij = (0,0)
        self.setup()
        self.cls()

    def cls(self):           # clear screen
        #self.can = Canvas([0,0,self.n+1,self.m+2])
        self.can = Canvas([0,0,self.n+1,self.m+1])
        return self.can

    def setup(self):
        self.r0 = 0.45;  self.r1 = 0.38;  self.r2 = 0.31
        self.r3 = 0.16;  self.rs = (self.r2-self.r3)*0.4

        self.ds = 0.11; self.rs = self.ds/3;
        self.gray = (0.8,0.8,0.8);  self.red = (1,0,0)
        self.gold = (1,0.9,0);      self.dark = (0.5,0.5,0.5)
        self.blue = (0,0.5,1);      self.green=(0,0.8,0)
        self.magenta = (1,0.2,1);   self.orange = (1,0.5,0)

    def basal(self,x,y,q):
        l = len(q)                     # number of basal synapses
        r2 = self.r2;  r3 = self.r3
        rs = self.rs
        rm = (r2+r3)/2                 # middle between r2 and r3
        xs = x

        dx = rs*1.2
        h = dx*1.414

        k = 0;
        if (l % 2 == 1):               # if an even number of basal synapses
            print("q:",q,"len(q):",len(q))
            ys = y+r2*1.25
            col = 'w' if q[k] == 0 else self.magenta;  k += 1
            self.can.circle((xs,ys),self.rs,col)
            n = int((l-1)/2)
            for i in range(1,n+1):
                col = 'w' if q[k] == 0 else self.magenta;  k += 1
                self.can.circle((xs-i*h,ys-i*h),rs,col)
                col = 'w' if q[k] == 0 else self.magenta;  k += 1
                self.can.circle((xs+i*h,ys-i*h),rs,col)
        else:
            xs = x;  ys = y+r2
            n = int(l/2)
            for i in range(n-1,-1,-1):
                col = 'w' if q[k] == 0 else self.magenta;  k += 1
                self.can.circle((xs-i*h-dx,ys-i*h),rs,col)
            for i in range(0,n):
                col = 'w' if q[k] == 0 else self.magenta;  k += 1
                self.can.circle((x+i*h+dx,ys-i*h),rs,col)

    def segment(self,x,y,r,d,mu,P,Q,L):  # plot mu-th dendritic segment out of total d
        H = r*0.9;  W = 2*r            # total height/width of all segments
        yoff = r*0.2                   # y offset of segments
        h = H/d; w = W/2               # height and half width of segment
        dy = r/4
        ymu = y + yoff - mu*h          # top position of mu-th segment


        learn = L[mu].any()
        col = self.gold if learn else self.gray

        xs = x;  ys = ymu-h/2
        self.can.fancy((x-w,ymu-h),(x+w,ymu),col,r=r/10)

        d0 = self.d-1;  s0 = self.s-1
        ws = min(h*0.4,w/self.s)
        self.rs = ws*0.8

        for nu in range(0,self.s):
            xs = x + 2*ws*nu - (self.s*ws/2 + 1.5*ws);
            yy = ys + h*(d0-mu)
            if Q[mu,nu] > 0:
                col = self.magenta
            else:
                col = 'w' if P[mu,nu] >= 0.5 else 'k'
            self.can.circle((xs,ys),self.rs,col)

    def neuron(self,ij,u=None,x=None,y=None,b=None,P=None,Q=None,L=None,q=None):
        u = u if u != None else 0      # basal input
        x = x if x != None else 0      # predictive state
        y = y if y != None else 0      # output state
        b = b if b != None else 0      # burst state

        #print("P:\n",P)
        P = np.random.rand(self.d,self.s) if P is None else P
        Q = P*0 if Q is None else Q    # permanence matrix
        L = P*0 if L is None else L    # learning matrix
        q = [0,0,0,0] if q is None else q

        colu = self.blue if u else self.gray
        colb = self.orange if b else self.gray
        colx = self.green if x>0 else self.dark
        coly = self.red if y>0 else self.gray

        i = ij[0];  j = ij[1]
#       x = 1+j; y = self.m+1-i;
        x = 1+j; y = self.m-i;

        r0 = self.r0;  r2 = self.r2;  r3 = self.r3
        dy1 = r0*0.1;    dy2 = r0*0.1

            # draw different parts of neuron cell

        self.can.fancy((x-r0*0.9,y-r0*1.0),(x+r0*0.9,y-r0*0.2),colb,r=0.2)
        #self.can.fancy((x-r3,y+r3),(x+r3,y+3*r3),colu,r=0.05,angle=45)

        self.can.fancy((x-r2,y+dy1-r2),(x+r2,y+dy1+r2),colu,r=0.05,angle=45)
        self.can.fancy((x-r3,y+dy1-r3),(x+r3,y+dy1+r3),colx,r=0.04,angle=45)
        self.can.fancy((x-r0*0.4,y-r0+dy2),(x+r0*0.4,y-r0*0.2+dy2),coly,r=0.05,angle=45)

            # draw dentritic segments

        d = self.d #+1
        for mu in range(0,d):
            self.segment(x,y,self.r0,d,mu,P,Q,L)

           # draw basal dendritic segment

        self.basal(x,y,q)


    def cell(self,ij,u=None,x=None,y=None,P=None,Q=None,L=None):
        u = u if u != None else 0      # basal input
        x = x if x != None else 0      # predictive state
        y = y if y != None else 0      # output state

        P = P if P != None else numpy.random.rand(self.d,self.s)
        Q = Q if Q != None else P*0    # permanence matrix
        L = L if L != None else P*0    # learning matrix

        outer = self.red if y>0 else self.gray
        inner = self.green if x>0 else self.dark
        core  = self.gold if L.any().any() else self.gray

        i = ij[0];  j = ij[1]
        x = 1+j; y = self.m+1-i;
        self.can.circle((x,y),self.r0,outer)
        self.can.circle((x,y),self.r1,inner)
        self.can.circle((x,y),self.r2,core)

        d0 = self.d-1;  s0 = self.s-1
        for mu in range(0,self.d):
            for nu in range(0,self.s):
                xx = x + self.ds*(nu-s0/2);
                yy = y + self.ds*(d0-mu-d0/2)
                if L[mu,nu] > 0 and P[mu,nu] < 0.5:
                    col = self.red
                elif L[mu,nu] > 0 and P[mu,nu] >= 0.5:
                    col = self.green
                elif Q[mu,nu] > 0:
                    col = self.magenta
                elif L[mu,nu] > 0 and P[mu,nu] < 0.5:
                    col = 'b'
                else:
                    col = 'w' if P[mu,nu] >= 0.5 else 'k'
                self.can.circle((xx,yy),self.rs,col)

    def input(self,j,u):
        return
        u = u if u != None else 1
        x = 1+j; y = 1;
        col = self.blue if u > 0 else self.gray
        self.can.circle((x,y),self.r2,col)

    def at(self,i,j):  # to tell a Cell constructor where to place a cell
        self.ij = (i,j)
        return self

    def show(self):
        plt.show()

#===============================================================================
# class Monitor
# usage: mon = Monitor(4,10)
#        cell = Cell(mon,k,g,K,P)
#        cell.show()
#===============================================================================

import numpy

class Monitor:
    def __init__(self,m,n,verbose=0):
        self.screen = Screen('Neurons',m,n)
        self.ij = (0,0)
        self.verbose = verbose
        self.iteration = 0
        self.phase = None

            # recorder init

        self.initrecorder()

    def initrecorder(self):
        nan = float('nan');

        self.c = None

        self._P = None
        self.x_ = None
        self.P_ = None

        self.s = nan            # dendritic spike
        self.q = [0,0,0,0]

        self.W = None           # no weights needed
        self.V = None           # no pre-synaptic signals needed
        self.Q = None           # no synaptics
        self.L = None           # no binary learning matrix
        self.D = None           # no binary learning matrix

    def record(self,cell,u,c,q=None,V=None,W=None,Q=None,L=None,D=None,s=None):
        self.c = cell.update(c);
        self.x_ = cell.x_;  self.P_ = cell.P_
        if q is None:
            self.log(cell,'(phase 1)',phase=1)
        elif W is None:
            self.q = q
            self.log(cell,"(phase 2)",phase=2)
        else:
            self.V = V;  self.W = W;  self.Q = Q;
            self.L = L;  self.D = D;  self.s = s;
            self.log(cell,"(phase 3)",phase=3)

    def place(self,screen,ij):
        self.screen = screen
        self.ij = ij

    def at(self,screen):
        if screen != None:
            self.place(screen,screen.ij)

    def plot(self,cell,i=None,j=None,q=None,Q=None):
        if i != None:
            self.place(self.screen,(i,j))
        self.q = self.q if q is None else q
        self.Q = self.Q if Q is None else Q

            # now copy elements of self.q except those element
            # which repsents our own cell output

        q = [];
        for k in range(0,len(self.q)):
             if cell.g[k] != cell.k:
                 q.append(self.q[k])

        self.screen.neuron(self.ij,cell.u,cell.x,cell.y,cell.b,
                           cell.P,self.Q,self.L,self.q)
        self.screen.input(self.ij[1],cell.u)
        self.screen.show

    def log(self,cell,msg=None,phase=None):
        nan = float('nan')
        msg = msg if msg != None else ""
        self.phase = phase if phase != None else self.phase
        print("-------------------------------------------------------------")
        print("iteration: ",self.iteration,"cell: #%g" % cell.k,msg)
        print("   k:",cell.k,", g:",cell.g,", eta:",cell.eta)
        self.print('matrix',"   K:",cell.K)
        self.print('matrix',"   P:",cell.P)
        if (self.phase == 3):
            self.print('matrix',"   W:",self.W)
            self.print('matrix',"   V:",self.V)
            self.print('matrix',"   Q:",self.Q)
            self.print('matrix',"   L:",self.L)
            self.print('matrix',"   D:",self.D)
        if (self.phase== 2 or self.phase == 3):
            print("   b:",cell.b,"(q:", self.q,
              ", ||q||=%g)" % (nan if numpy.isnan(self.q).any() else sum(self.q)))
        if (self.phase == 3):
            print("   s:",int(self.s),"(||Q||=%g, theta:%g)" % (norm(self.Q),cell.theta))
        print("   u:",cell.u)
        if (self.phase == 3):
            print("   x: %g (-> %g)" % (cell.x,cell.x_))
        else:
            print("   x: %g" % cell.x)
        print("   y: %g" % cell.y)
        print("   c:",self.c)
        print("-------------------------------------------------------------")

        if (phase == 3):
            #self.invalid(cell,'b,p,l,W,Z')       # invalidate
            self.iteration += 1

    def print(self,tag,msg,arg):   # .print("matrix","Q:",Q)
        if tag == 'matrix':
            m,n = arg.shape
            print(msg,"[",end='')
            sepi = ''
            for i in range(0,m):
                print(sepi,end='');  sepi = '; ';  sepj = ''
                for j in range(0,n):
                    s = "%4g" % arg[i,j].item()
                    s = s if s[0:2] != '0.' else s[1:]
                    s = s if s[0:3] != '-0.' else '-'+s[2:]
                    print("%5s" % s, end='');
                    sepj = ' '
            print(']')
        elif tag == 'number':
            print(msg,"%4g" % arg)

    def show(self,i=None,j=None):
        if i != None:
            self.plot(i,j)
        can = self.neurons.canvas()
        self.plot()

    def hello(self):
        print("hello, monitor")


#===============================================================================
# helper: matrix 1-norm (maximum of row sums)
#===============================================================================

def norm(M):    # max of row sums
    result = 0
    for j in range(0,M.shape[0]):
        sumj = M[j].sum().item()
        result = result if sumj < result else sumj
    return result
