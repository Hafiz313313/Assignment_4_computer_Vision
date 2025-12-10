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



# --- epipolar distance (pixel space) ---------------------------------------
def compute_epipolar_errors(F, x1s, x2s):
    """
    Compute distances from x2s to epipolar lines l2 = F * x1s.

    Parameters
    ----------
    F : (3,3) ndarray
        Fundamental matrix (pixel coordinates).
    x1s, x2s : (3,N) ndarray
        Homogeneous image points in pixel coordinates.

    Returns
    -------
    d : (N,) ndarray
        Distance in pixels from each x2s[:,i] to its corresponding epipolar line.
    """
    # epipolar lines in image 2 for points x1s
    l2 = F @ x1s  # 3 x N
    a = l2[0, :]
    b = l2[1, :]
    c = l2[2, :]
    x = x2s[0, :] / x2s[2, :]
    y = x2s[1, :] / x2s[2, :]
    denom = np.sqrt(a * a + b * b)
    # protect against zero denom (shouldn't happen with valid lines)
    denom[denom == 0] = 1e-16
    d = np.abs(a * x + b * y + c) / denom
    return d


# --- extract four P2 solutions from E --------------------------------------
def extract_P_from_E(E):
    """
    Given essential matrix E, return the 4 possible camera matrices P2 (3x4).
    P1 is assumed to be [I | 0].

    Returns
    -------
    P : ndarray of shape (4, 3, 4)
        P[i] is the i-th candidate 3x4 camera matrix.
    """
    U, S, Vt = np.linalg.svd(E)
    # enforce proper rotation: det(UVt) should be +1
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1,  0],
                  [1,  0,  0],
                  [0,  0,  1]], dtype=float)

    u3 = U[:, 2].reshape(3, 1)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    P = np.zeros((4, 3, 4))
    P[0, :, :] = np.hstack((R1,  u3))
    P[1, :, :] = np.hstack((R1, -u3))
    P[2, :, :] = np.hstack((R2,  u3))
    P[3, :, :] = np.hstack((R2, -u3))

    return P


# --- triangulate many points with DLT -------------------------------------
def triangulate_DLT(P1, P2, x1s, x2s):
    """
    Triangulate N points using linear DLT, returning homogeneous 4xN points.

    Parameters
    ----------
    P1, P2 : (3,4) ndarray
        Camera projection matrices.
    x1s, x2s : (3,N) ndarray
        Matching homogeneous image points (pixel or normalized; be consistent with P).

    Returns
    -------
    X : (4,N) ndarray
        Homogeneous 3D points.
    """
    N = x1s.shape[1]
    X = np.zeros((4, N))
    for i in range(N):
        u1 = x1s[0, i] / x1s[2, i]
        v1 = x1s[1, i] / x1s[2, i]
        u2 = x2s[0, i] / x2s[2, i]
        v2 = x2s[1, i] / x2s[2, i]

        A = np.zeros((4, 4))
        A[0, :] = u1 * P1[2, :] - P1[0, :]
        A[1, :] = v1 * P1[2, :] - P1[1, :]
        A[2, :] = u2 * P2[2, :] - P2[0, :]
        A[3, :] = v2 * P2[2, :] - P2[1, :]

        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1, :]
        X[:, i] = Xh / Xh[-1]
    return X


# --- camera helpers --------------------------------------------------------
def camera_center(P):
    """
    Compute camera center C (3,) from projection matrix P = [R | t]
    """
    R = P[:, :3]
    t = P[:, 3]
    C = -np.linalg.inv(R) @ t
    return C


def principal_axis(P):
    """
    Return camera principal axis (unit vector) in world coords.
    For P = [R | t], principal axis = R^T * [0,0,1]^T
    """
    R = P[:, :3]
    a = R.T @ np.array([0.0, 0.0, 1.0])
    a = a / np.linalg.norm(a)
    return a
