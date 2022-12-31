import numpy as np
from sklearn.decomposition import PCA

class MeshProcess():
    def __init__(self, x, y, z, dx, dy, dz):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def gen_mesh(self):
        min_x = self.x.min()
        max_x = self.x.max()
        min_y = self.y.min()
        max_y = self.y.max()
        yMesh, xMesh, zMesh = np.mgrid[min_y:max_y+0.1:2.0, min_x:max_x+0.1:2.0, 0:0+0.1]
        for i, j, k in zip(self.x, self.y, self.z):
            coord_idx = np.argwhere((xMesh==i) & (yMesh==j))[0]
            zMesh[coord_idx[0], coord_idx[1], coord_idx[2]] = k
            zMesh[ zMesh == 0] = np.nan
            
        return xMesh, yMesh, zMesh

    def gen_disp(self):
        xMesh, yMesh, _ = self.gen_mesh()
        xDisp = np.full_like(xMesh, np.nan, dtype=np.double)
        yDisp = np.full_like(xMesh, np.nan, dtype=np.double)
        zDisp = np.full_like(xMesh, np.nan, dtype=np.double)

        # fill the displacement array with values from txt files
        for x, y, dx, dy, dz in zip(self.x, self.y, self.dx, self.dy, self.dz):
            coord_idx = np.argwhere((xMesh==x) & (yMesh==y))[0]
            xDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dx 
            yDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dy 
            zDisp[coord_idx[0], coord_idx[1], coord_idx[2]] = dz 

        return xDisp, yDisp, zDisp


    def mesh_append(self, depth=2):
        xMesh, yMesh, zMesh = self.gen_mesh()
        xMesh_append = np.concatenate((xMesh, xMesh), axis=2)
        yMesh_append = np.concatenate((yMesh, yMesh), axis=2)
        zMesh_append = np.concatenate((zMesh, zMesh-2), axis=2)
        return xMesh_append, yMesh_append, zMesh_append

    def disp_append(self):
        xDisp, yDisp, zDisp = self.gen_disp()
        zeros = np.zeros_like(xDisp)
        xDisp_append = np.concatenate((xDisp, zeros), axis=2)
        yDisp_append = np.concatenate((yDisp, zeros), axis=2)
        zDisp_append = np.concatenate((zDisp, zeros), axis=2)
        return xDisp_append, yDisp_append, zDisp_append

    @classmethod
    def df_process(cls, xyz, dxyz):
        x = xyz['Easting']
        y = xyz['Northing']
        z = xyz['elev']
        dx = dxyz['dEasting']
        dy = dxyz['dNorthing']
        dz = dxyz['RASTERVALU']
        disp_field = cls(x, y , z, dx, dy, dz)
        return disp_field
        
class PCAProjecion(MeshProcess):
    def __init__(self, n_component):
        super(MeshProcess, self).__init__()
        self.n_component = n_component

    @property
    def eig_vectors(self, n_components=3):
        surface = np.vstack(self.x, self.y, self.z)
        pca = PCA(n_components=n_components)
        pca.fit(surface)
        return pca.components_

    def mesh_trans(self):
        xMesh, yMesh, zMesh = self.gen_mesh()
        points = np.vstack([xMesh.ravel(), yMesh.ravel(), zMesh.ravel()])
        points_trans = np.dot(points.T, self.eig_vectors.T)
        xMesh_trans = points_trans[:,0].reshape(xMesh.shape)
        yMesh_trans = points_trans[:,1].reshape(xMesh.shape)
        zMesh_trans = points_trans[:,2].reshape(xMesh.shape)
        return xMesh_trans, yMesh_trans, zMesh_trans

    def disp_trans(self):
        xDisp, yDisp, zDisp = self.gen_disp()
        disp = np.vstack([xDisp.ravel(), yDisp.ravel(), zDisp.ravel()])
        return np.dot(disp, self.eig_vectors.T)
