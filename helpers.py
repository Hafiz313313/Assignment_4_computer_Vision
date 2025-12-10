# For your convenience:
# Paste the required functions from previous assignments here.
import numpy as np



def estimate_F_DLT(x1s, x2s):
    x1 = x1s[0, :] / x1s[2, :]
    y1 = x1s[1, :] / x1s[2, :]
    x2 = x2s[0, :] / x2s[2, :]
    y2 = x2s[1, :] / x2s[2, :]

    N = x1s.shape[1]
    M = np.zeros((N, 9))
    M[:, 0] = x2 * x1
    M[:, 1] = x2 * y1
    M[:, 2] = x2
    M[:, 3] = y2 * x1
    M[:, 4] = y2 * y1
    M[:, 5] = y2
    M[:, 6] = x1
    M[:, 7] = y1
    M[:, 8] = 1.0

    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1, :]
    F = v.reshape(3, 3)
    return F, M, S[-1], np.linalg.norm(M @ v)


def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    # Your code here
    U, S, Vt = np.linalg.svd(E_approx)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E



def convert_E_to_F(E,K1,K2):
    '''
    A function that gives you a fundamental matrix from an essential matrix and the two calibration matrices
    E - Essential matrix (3x3)
    K1 - Calibration matrix for the first image (3x3)
    K2 - Calibration matrix for the second image (3x3)
    '''
    # Your code here
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F



def enforce_fundamental(F_approx):
    '''
    F_approx - Approximate Fundamental matrix (3x3)
    '''
    # Your code here
    U, S, Vt = np.linalg.svd(F_approx)

    S[-1] = 0.0

    F = U @ np.diag(S) @ Vt
    return F
