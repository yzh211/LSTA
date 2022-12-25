import pytest
import os, sys
import kinematics
import numpy as np
import pandas as pd

data_path = './Sarah_dat/datasets/Text_files/'
data = data_path + '2019_2020_bef_bottom.txt',

class TestKinematics:
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def test_disp(self):
        xMesh, yMesh = kinematics.gen_mesh(self.data)
        coord_ind = np.argwhere((xMesh==715332.125) & (yMesh==4323492.5))[0]
