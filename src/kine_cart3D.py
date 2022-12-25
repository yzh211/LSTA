import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator

class Kine_3d():
    def __init__(self, x, y, z, u, v, w):
        self.du_x = np.gradient(u, axis=1)     
        self.du_y = np.gradient(u, axis=0)     
        self.du_z = np.gradient(u, axis=2)     
        self.dv_x = np.gradient(v, axis=1)     
        self.dv_y = np.gradient(v, axis=0)     
        self.dv_z = np.gradient(v, axis=2)     
        self.dw_x = np.gradient(w, axis=1)     
        self.dw_y = np.gradient(w, axis=0)     
        self.dw_z = np.gradient(w, axis=2)     
        self.dx = np.gradient(x, axis=1)
        self.dy = np.gradient(y, axis=0)
        self.dz = np.gradient(z, axis=2)

    @property
    def grad(self):
        return [
            self.du_x/self.dx,
            self.du_y/self.dy,
            self.du_z/self.dz,
            self.dv_x/self.dx,
            self.dv_y/self.dy,
            self.dv_z/self.dz,
            self.dw_x/self.dx,
            self.dw_y/self.dy,
            self.dw_z/self.dz
        ]

    def divergence(self):
        return self.grad[0] + self.grad[4] + self.grad[8]

    def curl_z(self):
        return self.grad[1] - self.grad[3]

    def curl_y(self):
        return self.grad[2] - self.grad[6]

    def curl_x(self):
        return self.grad[7] - self.grad[5]

def gen_mesh(x, y, z):
    min_x = x.min()
    max_x = x.max()
    min_y = y.min()
    max_y = y.max()
    # min_z = df['z'].min()
    # max_z = df['z'].max()
    yMesh, xMesh, zMesh = np.mgrid[min_y:max_y+0.1:2.0, min_x:max_x+0.1:2.0, 0:0+0.1]
    for i, j, k in zip(x, y, z):
        coord_idx = np.argwhere((xMesh==i) & (yMesh==j))[0]
        zMesh[coord_idx[0], coord_idx[1], coord_idx[2]] = k
        zMesh[ zMesh == 0] = np.nan
        
    return xMesh, yMesh, zMesh

def gen_disp(df, disp, xMesh, yMesh):
    xDisp = np.full_like(xMesh, np.nan, dtype=np.double)
    yDisp = np.full_like(xMesh, np.nan, dtype=np.double)
    zDisp = np.full_like(xMesh, np.nan, dtype=np.double)

    # fill the displacement array with values from txt files
    for x, y, dx, dy, dz in zip(df['Easting'], df['Northing'], disp['dEasting'], disp['dNorthing'], disp['RASTERVALU']):
        coord_idx = np.argwhere((xMesh==x) & (yMesh==y))[0]
        xDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dx 
        yDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dy 
        zDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dz 

    return xDisp, yDisp, zDisp

def grad_smooth(xMesh, yMesh, grad):
    points = np.stack([xMesh.ravel(), yMesh.ravel()], -1)
    values = grad.ravel()
    f = griddata(points, values, (xMesh, yMesh), method='nearest') 
    return f

def contour_plot(xMesh, yMesh, grad, saveName, ax=None, figsize=(3.5, 3.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    surf = ax.pcolormesh(xMesh, yMesh, grad, cmap='jet')
    plt.savefig("./pics/" + saveName, dpi=600)
    plt.close(fig)