"""
Module for basic creating and manipulating of second order tensors
"""

import os
import numpy as np
import itertools
from scipy.linalg import polar
from scipy.spatial.transform import Rotation as R

class SecondTensor(np.ndarray):
    """
    Base class for tensor creation and manipulation.
    """
    def __new__(cls, input_array, check_rank=None):
        obj = np.asarray(input_array).view(cls)
        obj.rank = len(obj.shape)

        if check_rank and check_rank != obj.rank:
            raise ValueError(f"{type(obj).__name__} input must be rank {check_rank}")

        return obj
    
    def __repr__(self):
        return f"{type(self).__name__}({self})"

    @property
    def symmetrized(self):
        """Returns a generally symmetrized tensor,
        calculated by taking the sum of tensor and
        its transpose with respect to all possible 
        permutations of indices

        :return: symmetrized tensor
        :rtype: ndarray
        """
        perms = list(itertools.permutations(range(self.rank)))
        return sum(np.transpose(self, ind) for ind in perms) / len(perms)

    def is_symmetric(self, tol:float = 1e-5):
        """Tests wether a tensor is symmetric or not

        :param tol: tolerance to test for symmetry
        :type tol: float, optional
        """
        return (self - self.symmetrized < tol).all()

    @property
    def trans(self):
        return SecondTensor(np.transpose(self))

    @property
    def inv(self):
        if self.det == 0:
            raise ValueError("Tensor is non-invertible")
        return SecondTensor(np.linalg.inv(self))

    @property
    def det(self):
        return np.linalg.det(self)

    @property
    def principal_invariants(self):
        """Returns a list of principal invariants for the tensor,
        which are the values of the coefficients of the characteristic
        polynomial for the matrix
        """
        return np.poly(self)[1:] * np.array([-1, 1, -1])

    @property
    def polar_decomposition(self, side="right"):
        """Return rotation matrix.

        :param side: right or left polar decomposition, defaults to "right"
        :type side: str, optional
        :return: rotation matrix
        """
        return polar(self, side=side)[0]

    @property
    def euler_angles(self, seq="zyx"):
        """Return Euler angles, following z-y-x order

        :param seq: sequence order, defaults to "zyx"
        :type seq: str, optional
        :return: euler_angles as dgrees
        """
        M = R.from_matrix(self.polar_decomposition)
        return M.as_euler(seq=seq, degrees=True)

    @property
    def green_lagrangian(self):
        """Return Green-Lagrangian Tensor
        """
        return 0.5 * (np.dot(self.trans, self) - np.eye(3)) 

    @property
    def angle_increment(self):
        """Angle changes between initially parallel axes.

        :return: angle changes between XY, XZ, and YZ
        :rtype: list
        """
        E = self.green_lagrangian
        theta_XY = -np.arcsin(2*E[0][1] / np.sqrt(1+2*E[0][0]) / np.sqrt(1+2*np.sqrt(1+2*E[1][1])))
        theta_XZ = -np.arcsin(2*E[0][2] / np.sqrt(1+2*E[0][0]) / np.sqrt(1+2*np.sqrt(1+2*E[2][2])))
        theta_YZ = -np.arcsin(2*E[1][2] / np.sqrt(1+2*E[1][1]) / np.sqrt(1+2*np.sqrt(1+2*E[2][2])))
        return [theta_XY, theta_XZ, theta_YZ]

    @property
    def stretching(self):
        """Stretching of material along each axis

        :rtype: list
        """
        E = self.green_lagrangian
        lambda_1 = np.sqrt(1 + 2*E[0][0]) 
        lambda_2 = np.sqrt(1 + 2*E[1][1]) 
        lambda_3 = np.sqrt(1 + 2*E[2][2]) 
        return [lambda_1, lambda_2, lambda_3]

if __name__ == "__main__":
    test = SecondTensor(
        input_array=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    )

    F = SecondTensor(
        [[1, -0.333, 0.959],
        [0.495, 1, 0],
        [0.5, -0.247, 1.5]]
    )

    F1 = np.asarray([
        [1, 0.495, 0.5],
        [-0.3333, 1, -0.247],
        [0.959, 0, 1.5]
    ])
     
    # M = R.from_matrix(F.polar_decomposition()[0])

    Q = np.asarray(
        [
            [1, 0, 0],
            [0, 1.732/2, 1/2],
            [0, -1/2, 1.732/2]
        ]
    )

    # M = R.from_matrix(Q)
    M = SecondTensor.green_lagrangian(F1)
    # N = SecondTensor(F1)
    # M = N.green_lagrangian()
    print(M)
   
