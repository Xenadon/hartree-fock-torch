import torch

def eigen_sqrt_nosym(B):
    '''
    Suppose B can be decomposed to B = U @ diag(S) @ U.T,
    returns X = U @ diag(S)^(-1/2), that satisfies X.T @ B @ X = I
    '''
    S, U = torch.linalg.eigh(B)
    S_pinv_sqrt = S.pow_(-1).sqrt_()
    return U * S_pinv_sqrt.reshape(B.shape[0],1,-1)

def transformed_eigh(A, X):
    '''
    returns the eigenvalue and (X @ eigenvectors) of X.T @ A @ X
    '''
    A_ = X.transpose(-1,-2) @ A @ X
    S_A_ , U_A_ = torch.linalg.eigh(A_)
    return S_A_, X @ U_A_  

def g_eigh(A, B):
    '''
    requires: B is nonsigular (numerically well-conditioned).

    returns: S, U: that satisfies 
        A @ U == B @ U @ diag(S)
    and 
        U.T @ B @ U == I.
    '''
    b = A.shape[0]
    X = eigen_sqrt_nosym(B)
    return transformed_eigh(A, X)