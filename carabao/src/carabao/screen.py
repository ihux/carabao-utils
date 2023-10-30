#===============================================================================
# carabao/screen library:
# - class Screen (copyright: Neuronycs 2023)
#===============================================================================

from numpy import array, isnan, copy, arange, zeros
from numpy.random import rand
from ypstruct import struct

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.transforms import Affine2D

#=============================================================================
# hash function
#=============================================================================

def hash(o,fac=1):
    """
    hash(): map list or matrix to an integer hash value

               g = [0,1,2]
               h = hash(g)

               P = numpy.array([[.1,.2,.3],[.4,.5,.6]])
               h = hash(P)

               h = hash(None)         # => h = 0
    """
    def weight(M):
        return 1+fac*10*(1 + arange(M.flatten().shape[0]))

    if o is None:
        return fac + 0
    elif type(o).__name__ == 'ndarray':
        f = o.flatten();   w = weight(o)
        #return (sum(f*w),f*w,f,w)
        return 1 + fac*int(sum(f*w))
    elif type(o).__name__ == 'list':
        return 1 + fac*hash(array(o))
    else:
        return 1 + fac*o
    return 0

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
    def frame(self):                   # draw frame
        xl = self.position[0];  xh = self.position[2]
        yl = self.position[1];  yh = self.position[3]-1
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
#        E = (P >= 0.5)                # synaptics
#        scr.plot((i,j),x,y,P,E)
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
    def cls(self):                     # clear screen
        #self.can = Canvas([0,0,self.n+1,self.m+2])
        self.can = Canvas([-1,-1,self.n,self.m+1])
        return self.can
    def setup(self):
        kr = 0.9
        self.r0 = kr*0.45;  self.r1 = kr*0.36;  self.r2 = kr*0.31
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
            #print("q:",q,"len(q):",len(q))
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
    def segment(self,x,y,r,d,mu,W,E,s):  # plot mu-th dend-seg out of total d
        H = r*0.9;                     # total height/width of all segments
        yoff = r*0.2                   # y offset of segments
        h = H/d; w = r                 # height and half width of segment
        dy = r/4
        ymu = y + yoff - mu*h          # top position of mu-th segment

        ################
        #print("segment: s =",s)
        col = self.gold if s[mu] > 0 else self.gray

        xs = x;  ys = ymu-h/2
        self.can.fancy((x-w,ymu-h),(x+w,ymu),col,r=r/10)

        d0 = self.d-1;  s0 = self.s-1
        ws = min(h*0.4,w/self.s)
        self.rs = ws*0.8

        for nu in range(0,W.shape[1]):
            xs = x + 2*ws*nu - (self.s*ws/2 + 1.5*ws);
            yy = ys + h*(d0-mu)
            if E[mu,nu] > 0:
                col = self.magenta
            else:
                col = 'w' if W[mu,nu] > 0 else 'k'
            self.can.circle((xs,ys),self.rs,col)
    def neuron(self,ij,u=None,x=None,y=None,b=None,v=None,s=None,W=None,E=None):
        u = u if u != None else 0      # basal input
        x = x if x != None else 0      # predictive state
        y = y if y != None else 0      # output state
        b = b if b != None else 0      # burst state

        v = [0,0,0,0] if v is None else v
        W = zeros((self.d,self.s)) if W is None else W
        E = W*0 if E is None else E    # permanence matrix
        s = [0 for k in range(0,E.shape[0])] if s is None else s

        colu = self.blue if u else self.gray
        colb = self.orange if b else self.gray
        colx = self.green if x>0 else self.dark
        coly = self.red if y>0 else self.gray

        i = ij[0];  j = ij[1]
        x = j; y = self.m-i-1;

        r0 = self.r0;  r2 = self.r2;  r3 = self.r3
        dy1 = r0*0.1;    dy2 = r0*0.1

            # draw different parts of neuron cell

        self.can.fancy((x-r0*0.9,y-r0*1.0),(x+r0*0.9,y-r0*0.2),colb,r=0.2)
        self.can.fancy((x-r2,y+dy1-r2),(x+r2,y+dy1+r2),colu,r=0.05,angle=45)
        self.can.fancy((x-r3,y+dy1-r3),(x+r3,y+dy1+r3),colx,r=0.04,angle=45)
        self.can.fancy((x-r0*0.4,y-r0+dy2),(x+r0*0.4,y-r0*0.2+dy2),coly,r=0.05,angle=45)

            # draw dentritic segments

        d = self.d #+1
        for mu in range(0,d):
            self.segment(x,y,self.r0,d,mu,W,E,s)

           # draw basal dendritic segment

        self.basal(x,y,v)
    def cell(self,ij,u=None,x=None,y=None,P=None,E=None,L=None):
        u = u if u is not None else 0      # basal input
        x = x if x is not None else 0      # predictive state
        y = y if y is not None else 0      # output state

        P = P if P is not None else rand(self.d,self.s)
        E = E if E is not None else P*0    # permanence matrix
        L = L if L is not None else P*0    # learning matrix

        outer = self.red if y>0 else self.gray
        inner = self.green if x>0 else self.dark
        core  = self.gold if L.any().any() else self.gray

        i = ij[0];  j = ij[1]
#>>>>>>>>>>>>>>>>>>>>>>>
        x = j; y = self.m-i;
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
                elif E[mu,nu] > 0:
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

class Monitor:
    data = struct()
    def __init__(self,m=None,n=None,verbose=0):
        if m is not None:
            data = self.data
            data.screen = Screen('Neurons',m,n)
            data.ij = (0,0)
            data.verbose = verbose
            data.iteration = 0
            data.phase = None
    def copy(self):
        scr = self.data.screen
        mon = Monitor()
        mon.data = self.data.copy()
        return mon
    def place(self,screen,ij):
        self.data.screen = screen
        self.data.ij = ij
    def plot(self,cell,i=None,j=None,v=None,W=None,E=None):
       data = self.data;  aux = cell.aux
       if i is not None:
            self.place(data.screen,(i,j))
            c = [] if aux.c is None else aux.c;
            W = cell.W() if W is None else W
            E = cell.E(c) if E is None else E
            s = cell.s(c)
            data.screen.neuron(data.ij,aux.u,cell.x,cell.y,cell.b,
                               data.v,s,W,E)
            #data.screen.input(data.ij[1],aux.u)
            data.screen.show
    def norm1(self,M):
        if type(M).__name__ == 'list':
            return sum(M)

        result = 0
        for j in range(0,M.shape[0]):
            sumj = M[j].sum().item()
            result = result if sumj < result else sumj
        return result
    def log(self,cell,msg=None,phase=None):
        always = True
        k = cell.k
        c = cell.aux.c
        s = cell.s(c)
        data = self.data;  aux = cell.aux
        nan = float('nan')
        msg = msg if msg != None else ""
        data.phase = phase if phase != None else data.phase
        print("--------------------------------------------------------------")
        print("iteration: ",data.iteration,"cell: #%g" % cell.k,msg)
        print("   k%g:" % k,cell.k,", g:",cell.g,", eta:",cell.eta)
        self.print('matrix',"   K%g:" % k,cell.K)
        self.print('matrix',"   P%g:" % k,cell.P)
        if (always or data.phase == 3):
            self.print('matrix',"   V%g:" % k, aux.V)
            self.print('matrix',"   W%g:" % k, aux.W)
            self.print('matrix',"   E%g:" % k, aux.E)
            self.print('matrix',"   L%g:" % k, aux.L)
            self.print('matrix',"   D%g:" % k, aux.D)
        if (always or data.phase== 2 or data.phase == 3):
            #print("   b%g:" % k,cell.b,"(q%g:" % k, data.v,
            #  ", ||v%g||=%g)" % (k,nan if isnan(data.v).any() else sum(data.v)))
            print("   b%g:" % k,cell.b,", v%g:" % k, cell.v(c))
        if (always or data.phase == 3):
            print("   s%g:" % k, s,"(||E||=%g, theta:%g)" % (self.norm1(aux.E),cell.theta))
            print("   u%g:"%k,aux.u,", y%g: %g" % (k,cell.y),", x%g: %g (-> %g)" % (k,cell.x,cell.x_))
        else:
            print("   u%g:"%k,aux.u,", y%g: %g" % (k,cell.y),", x%g: %g" % (k,cell.x))
        print("   c:",aux.c)
        print("-------------------------------------------------------------")

        if (phase == 3):
            #self.invalid(cell,'b,p,l,W,Z')       # invalidate
            self.data.iteration += 1
    def print(self,tag,msg,arg):   # .print("matrix","E:",E)
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
    def hash(self,cell):
        data = self.data;  aux = cell.aux
        v = cell.v(aux.c)
        s = cell.s(aux.c)
        hk = hash(cell.k,2);  hg = hash(cell.g,3);
        hK = hash(cell.K,4);  hP = hash(cell.P,5);
        hu = hash(aux.u,5);   hx = hash(cell.x,6);  hy = hash(cell.y,7);
        hs = hash(s,8);  hb = hash(cell.b,9)
        hq = hash(v,10)
        hW = hash(aux.W,11);  hV = hash(aux.V,12);  hE = hash(aux.E,13)
        hL = hash(aux.L,14);  hD = hash(aux.D,15)

        hashes = [[hk,hg,hK,hP],[hu,hx,hy,hs,hb],[hq,hW,hV,hE,hL,hD]]
        prime = 1*2*3*5*7*11*13*17*19+1
        N = (1 + hk*hg*hk*hP * hu*hx*hy*hs*hb + hq*hW*hV*hE*hL*hD)
        n = N % prime
        #return (h,N,prime,hashes)
        return n
    def ascii(self,n):  # convert hash to 4 character ascii sequence
        vocal = ['A','E','I','O','U','Y']
        h = ''
        for i in range(0,2):
           h = h + chr(65 + n % 26);  n = n // 26;
           k = n % 6;  n = n // 6;  h += vocal[k]
        return h
    def line(self,x,y,color='k',linewidth=0.5):
        plt.plot(x,y,color,linewidth=linewidth)
    def text(self,x,y,txt,color='k',size=None,rotation=0,ha='center',va='center'):
        size = 10 if size is None else size
        plt.text(x,y, txt, size=size, rotation=rotation, ha=ha, va=va, color=color)
    def separator(self,j,color='k',linewidth=0.5):
        scr = self.data.screen
        self.line([j-0.5,j-0.5],[-1,scr.m],color=color,linewidth=linewidth)
    def xlabel(self,x,txt,size=None):
        self.text(x,-0.75,txt)
    def head(self,txt,size=7):
        ij = self.data.ij
        scr = self.data.screen
        y = scr.m-ij[0]-2
        self.text(ij[1],y+1.65,txt,size=size)
    def foot(self,txt,size=7):
        ij = self.data.ij
        scr = self.data.screen
        y = scr.m-ij[0]-2
        self.text(ij[1],y+0.35,txt,size=size)
