import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def dat_combine(datList):
    """Combine all parts of landslides

    :param datList: list contains names of parts
    :type datList: list
    :return: dataframe contains data of landslides
    :rtype: dataframe
    """
    df = pd.DataFrame()
    for dat in datList:
        df_tmp = pd.read_csv(dat)
        df = df.append(df_tmp)

    return df

class Kine_2d():
    def __init__(self, x, y, u, v):
        self.du_x = np.gradient(u, axis=1)     
        self.du_y = np.gradient(u, axis=0)     
        self.dv_x = np.gradient(v, axis=1)
        self.dv_y = np.gradient(v, axis=0)
        self.dx = np.gradient(x, axis=1)
        self.dy = np.gradient(y, axis=0)

    @property
    def grad(self):
        return [
            self.du_x/self.dx,
            self.du_y/self.dy,
            self.dv_x/self.dx,
            self.dv_y/self.dy
        ]

    def divergence(self):
        return self.grad[0] + self.grad[3]

    def curl(self):
        return self.grad[1] - self.grad[2]

def gen_mesh(df):
    min_x = df['Easting'].min()
    max_x = df['Easting'].max()
    min_y = df['Northing'].min()
    max_y = df['Northing'].max()
    yMesh, xMesh = np.mgrid[min_y:max_y+0.1:2.0, min_x:max_x+0.1:2.0]
    return xMesh, yMesh

def gen_disp(df, xMesh, yMesh):
    xDisp = np.full_like(xMesh, np.nan, dtype=np.double)
    yDisp = np.full_like(xMesh, np.nan, dtype=np.double)

    # fill the displacement array with values from txt files
    for x, y, dx, dy in zip(df['Easting'], df['Northing'], df['dEasting'], df['dNorthing']):
        coord_idx = np.argwhere((xMesh==x) & (yMesh==y))[0]
        xDisp[coord_idx[0], coord_idx[1]] = dx 
        yDisp[coord_idx[0], coord_idx[1]] = dy 

    return xDisp, yDisp
    
if __name__ == "__main__":
    data_path = './Sarah_dat/datasets/Text_files/'
    dataList = [
        data_path + '2019_2020_bef_bottom.txt',
        data_path + '2019_2020_bef_lowandno.txt',
        data_path + '2019_2020_bef_outside.txt',
        data_path + '2019_2020_bef_top.txt'
    ]

    df = dat_combine(dataList)
    xMesh, yMesh = gen_mesh(df)
    xDisp, yDisp = gen_disp(df, xMesh, yMesh)

    kinematics = Kinematics_2d(xMesh, yMesh, xDisp, yDisp)