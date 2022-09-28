import numpy as np
import os
from scipy.special import binom
import matplotlib.pyplot as plt
from shutil import rmtree

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def bernstein(self, n, k, t):
        return binom(n,k) * t**k * (1.-t)**(n-k)

    def bezier(self, points, num=200):
        N = len(points)
        t = np.linspace(0,1,num=num)
        curve = np.zeros((num,2))
        for i in range(N):
            curve += np.outer(self.bernstein(N-1,i,t), points[i])
        return curve

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = self.bezier(self.p,self.numpoints)

class ContourGenerator:
    def __init__(self, radius, edge):
        self.radius = radius
        self.edge = edge

    @staticmethod
    def ccw_sort(p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def get_curve(self, points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def get_random_points(self, n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(ContourGenerator.ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            return a*scale
        else:
            return self.get_random_points(n,scale,mindst,rec+1)

    def get_bezier_curve(self, base_points):
        """ given an array of points, create a curve through
        those points.
        *rad* is a number between 0 and 1 to steer the distance of
              control points.
        *edge* is a parameter which controls how "edge" the curve is,
               edge=0 is smoothest."""

        p = np.arctan(self.edge)/np.pi+.5
        points = ContourGenerator.ccw_sort(base_points)
        points = np.append(points,np.atleast_2d(points[0,:]),axis=0)
        d = np.diff(points,axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        points = np.append(points,np.atleast_2d(ang).T,axis=1)
        s, c = self.get_curve(points,r=self.radius,method="var")
        x,y = c.T

        return x,y

def generate_contours(n=3, m=1, rmax=1, emax=1, export_dir=os.getcwd()):

    if os.path.exists(export_dir):
        rmtree(export_dir)
    os.makedirs(export_dir)

    print('Generating contours...')
    print('   Contours generated:')
    for i in range(m):
        r = rmax * np.random.random()
        edge = emax * np.random.random()
        generator = ContourGenerator(r,edge)
        base_points = generator.get_random_points(n,scale=1)
        x,y = generator.get_bezier_curve(base_points)

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        plt.plot(x,y,color='k',linewidth=2)
        plt.axis('off')
        fig.savefig(os.path.join(export_dir,'rmax={}_emax={}_n={}_contour_{}.png'.format(rmax,emax,n,(i+1))), dpi=100)
        plt.close()
        print('     {}/{}'.format((i+1),m))

def rename_files(rmax, emax, n, contours_dir, format='png'):

    def get_contour_idx(name):
        idx_1 = name.find('contour')
        idx_2 = name[idx_1:].split('_')[-1].split('.')[0]

        return int(idx_2)

    files = [file for file in os.listdir(contours_dir) if file.endswith(format)]
    files.sort(key=get_contour_idx)
    filepaths = [os.path.join(contours_dir,file) for file in files]
    new_filepaths = [os.path.join(contours_dir,'rmax={}_emax={}_n={}_contour_{}.{}'.format(rmax,emax,n,(i+1),format))
                     for i in range(len(files))]
    for i in range(len(filepaths)):
        os.rename(filepaths[i],new_filepaths[i])


if __name__ == "__main__":
# *rad, the radius around the points at which the control points of the bezier curve sit. 
# This number is relative to the distance between adjacent points and should hence be 
# between 0 and 1. The larger the radius, the sharper the features of the curve.
# *edge, a parameter to determine the smoothness of the curve. If 0 the angle of the curve 
# through each point will be the mean between the direction to adjacent points. The larger 
# it gets, the more the angle will be determined only by one adjacent point. The curve 
# hence gets "edgier".
# *n the number of random points to use. Of course the minimum number of points is 3. 
# The more points you use, the more feature rich the shapes can become; at the risk of 
# creating overlaps or loops in the curve.

    export_dir = r'C:\Users\juan.ramos\Contour_generator\Datasets'
    m = 100
    n = 4
    rmax = 0.5
    emax = 0.05
    generate_contours(n,m,rmax,emax,export_dir)
    rmax_2 = 1
    emax_2 = 0.75
    n_2 = 7
    contours_dir = r'C:\Users\juan.ramos\Contour_generator\Datasets_rmax={}_emax={}_n={}'.format(rmax_2,emax_2,n_2)
    #rename_files(rmax_2,emax_2,n_2,contours_dir)
    print