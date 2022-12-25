import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kine_cart2D import dat_combine
from sklearn.decomposition import PCA
from kine_cart3D import gen_mesh, gen_disp, Kine_3d, contour_plot, grad_smooth

def sort_data(data_path):
    years = ['2019_2020', '2020_2021', '2021_2022']
    dat_area = ["_bef_bottom.txt", "_bef_lowandno.txt", "_bef_top.txt"]
    data_2019, data_2020, data_2021 = ([] for i in range(3))
    data_list = [data_2019, data_2020, data_2021]

    for ind, year in enumerate(years):
        for area in dat_area:
            data_list[ind].append(data_path+year+area)

    return data_list

def mesh_trans(xMesh, yMesh, zMesh, eig_vec):
    # orig_size = xMesh.shape
    points = np.vstack([xMesh.ravel(), yMesh.ravel(), zMesh.ravel()])
    points_trans = np.dot(points.T, eig_vec.T)
    xMesh_trans = points_trans[:,0].reshape(xMesh.shape)
    yMesh_trans = points_trans[:,1].reshape(xMesh.shape)
    zMesh_trans = points_trans[:,2].reshape(xMesh.shape)
    return xMesh_trans, yMesh_trans, zMesh_trans

def disp_trans(disp, eig_vec):
    return np.dot(disp, eig_vec.T)

def mesh_append(xMesh, yMesh, zMesh, depth=2):
    xMesh_append = np.concatenate((xMesh, xMesh), axis=2)
    yMesh_append = np.concatenate((yMesh, yMesh), axis=2)
    zMesh_append = np.concatenate((zMesh, zMesh-2), axis=2)
    return xMesh_append, yMesh_append, zMesh_append

def disp_append(xDisp, yDisp, zDisp):
    zeros = np.zeros_like(xDisp)
    xDisp_append = np.concatenate((xDisp, zeros), axis=2)
    yDisp_append = np.concatenate((yDisp, zeros), axis=2)
    zDisp_append = np.concatenate((zDisp, zeros), axis=2)
    return xDisp_append, yDisp_append, zDisp_append

def eig_vectors(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.components_


def data_trans(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    # coord_transform = pca.transform(coord)
    # disp_transform = pca.transform(disp)
    eig_vec = pca.components_
    coord_transform = np.dot(data, eig_vec.T)
    # data_trans = np.matmul(eig_vec, data.T)
    # return coord_transform,data_trans

# def addBed_vol(data, depth):
#     bed_data = data.eval("elev = elev - 2", inplace=True)
#     return bed_data
    

    # normal vector and centroid
    # normal_vec = eig_vec[2,:]
    # centroid = np.mean(data, axis=0)
    # d = -centroid.dot(normal_vec)

    # draw plane
    # xx, yy = np.meshgrid(
    #     np.linspace(data['Easting'].min(), data['Easting'].max(), 10),
    #     np.linspace(data['Northing'].min(), data['Northing'].max(), 10),
    #     )
    # z = (-normal_vec[0]*xx - normal_vec[1]*yy - d) * 1. / normal_vec[2]

    # plot the plane
    plt3d = plt.figure(figsize=(3.5,3.5)).gca(projection='3d')
    # plt3d.plot_surface(xx, yy, z, alpha=0.5)
    # plt3d.scatter(data['easting'], data['northing'], data['elev'], s=0.5, alpha=1, color='k')
    plt3d.scatter(coord_transform[:,0], coord_transform[:,1], coord_transform[:,2] ,s=0.5, alpha=1, color='k')
    plt.savefig('./pics/test2.jpg',dpi=1000)

            
if __name__ == "__main__":
    data_path = './Sarah_dat/datasets/Taylor_2021_2022/COsi_v4/'
    data_list = sort_data(data_path)

    i = 0
    for data in data_list:
        df = dat_combine(data)
        df_xyz = df[['Easting', 'Northing', 'elev']]
        df_dxyz = df[['dEasting', 'dNorthing', 'RASTERVALU']]
        xMesh, yMesh, zMesh = gen_mesh(df_xyz['Easting'], df_xyz['Northing'], df_xyz['elev'])
        xDisp, yDisp, zDisp = gen_disp(df_xyz, df_dxyz, xMesh, yMesh)

        # Transform the mesh and disp
        eig_vec = eig_vectors(df_xyz)
        xMesh_pca, yMesh_pca, zMesh_pca = mesh_trans(
            xMesh,
            yMesh,
            zMesh,
            eig_vec
        )

        disp_pca = disp_trans(df_dxyz, eig_vec)

        # append bedrock mesh
        xMesh_apd, yMesh_apd, zMesh_apd = mesh_append(
            xMesh_pca,
            yMesh_pca,
            zMesh_pca
        )
        xDisp_apd, yDisp_apd, zDisp_apd = disp_append(
            xDisp,
            yDisp,
            zDisp
        )

        # kinematics
        kinematics = Kine_3d(
            xMesh_apd, 
            yMesh_apd, 
            zMesh_apd, 
            xDisp_apd,
            yDisp_apd,
            zDisp_apd
        )

        div = kinematics.divergence()
        curl_z = kinematics.curl_z()
        curl_smooth = grad_smooth(xMesh[:,:,0], yMesh[:,:,0], div[:,:,0])

        save_name = ['cart3D_div_2019.jpg','cart3D_div_2020.jpg', 'cart3D_div_2021.jpg' ] 
        contour_plot(xMesh[:,:,0], yMesh[:,:,0], curl_smooth, save_name[i])
        i += 1

        # data_trans(df_dxyz)
        # coord_trans, disp_trans = data_trans(df_xyz)
        # print(coord_trans[0:5,:])
        # print(disp_trans[:,0:5])


        
