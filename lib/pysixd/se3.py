import numpy as np
import numpy.matlib as npm
from transforms3d.quaternions import mat2quat, quat2mat, quat2axangle
import scipy.stats as sci_stats


# RT is a 3x4 matrix
def se3_inverse(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3, 1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape(3)
    return RT_new


def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape(3)
    return RT_new


def T_inv_transform(T_src, T_tgt):
    """
    :param T_src:
    :param T_tgt:
    :return: T_delta: delta in pixel
    """
    T_delta = np.zeros((3,), dtype=np.float32)

    T_delta[0] = T_tgt[0] / T_tgt[2] - T_src[0] / T_src[2]
    T_delta[1] = T_tgt[1] / T_tgt[2] - T_src[1] / T_src[2]
    T_delta[2] = np.log(T_src[2] / T_tgt[2])

    return T_delta


def rotation_x(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1
    R[1, 1] = np.cos(t)
    R[1, 2] = -(np.sin(t))
    R[2, 1] = np.sin(t)
    R[2, 2] = np.cos(t)
    return R


def rotation_y(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 2] = np.sin(t)
    R[1, 1] = 1
    R[2, 0] = -(np.sin(t))
    R[2, 2] = np.cos(t)
    return R


def rotation_z(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 1] = -(np.sin(t))
    R[1, 0] = np.sin(t)
    R[1, 1] = np.cos(t)
    R[2, 2] = 1
    return R


def angular_distance(quat):
    vec, theta = quat2axangle(quat)
    return theta / np.pi * 180


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # multiply q with its transposed version q' and add A
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)
