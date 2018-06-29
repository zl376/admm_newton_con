import numpy as np
from utils import *


N_ITER_SHOW = 1
MARGIN = 1E-4


def CG(ndim, b, A, x_init=None, tol=1E-6, max_iter=100, verbose=True):
    """
    Input:
        ndim,           int                     dimension of unknown
        b,              ndarray (ndim,)         RHS
        A,              func_obj                LHS system matrix function 
        x_init,         ndarray (ndim,)         init. guess of unknown
        tol,            float                   stop. criterion:    |r|^2/|r0|^2 <= tol
        max_iter,       int                     stop. criterion:    n_iter == max_iter
        verbose,        bool
    Output:
        x,              ndarray (ndim,)
        n_iter,         int
        norm_rel_r,     float                   relative residual norm |r|^2/|r0|^2
    Note:
        Conjugate Gradient solving
            Ax = b
        or equivalently, 
            min {1/2 |Ax-b|^2}
    """
    # Initial guess
    if x_init is None:
        x_init = np.zeros(ndim)

    # loop for CG
    n_iter = 0
    x = x_init
    r0 = b - A(x)
    r = r0
    p = r
    while n_iter < max_iter and np.dot(r.T, r)/np.dot(r0.T, r0) > tol:
        
        # save old variable
        x_prev = x.copy()
        
        norm_r = np.dot(r.T, r)
        
        # calc. A(p)
        Ap = A(p)
        
        # calc. step size
        alpha = norm_r / np.dot(p.T, Ap)

        # update x
        x = x + alpha * p
        
        # update residual
        r = r - alpha * Ap
            
        # update search direction p
        beta = np.dot(r.T, r) / norm_r
        p = r + beta * p

        n_iter += 1
        
        # plot
        if verbose and n_iter%N_ITER_SHOW == 0:
            print('Iter {0}, |r|/|r0| {1:.3e}'.format(n_iter, np.dot(r.T, r)))

    return x, n_iter, np.dot(r.T, r)/np.dot(b.T, b)


def ADMM_ineqcons(ndim, func_cost, func_g_cost, func_h_cost=None, A=None, x_init=None, rho=10, tol=1E-6, max_iter=100, solver='direct', solver_param={}, verbose=True):
    """
    Input:
        ndim,           int                     dimension of unknown
        func_cost,      func_obj                cost function
        func_g_cost,    func_obj                grad. of cost function
        func_h_cost,    func_obj                hessian of cost function
        A,              ndarray (naux, ndim)    A * x; If no constraint, set to None
        x_init,         ndarray (ndim,)         init. guess of unknown
        rho,            float                   step size in dual ascent
        tol,            float                   stop. criterion:    |dx|^2/|x|^2 <= tol
        max_iter,       int                     stop. criterion:    n_iter == max_iter
        solver,         str                     method for Newton update: {'direct', 'CG'}
        solver_param,   dict                    parameter for Newton update
        verbose,        bool
    Output:
        x,              ndarray (ndim,)
        n_iter,         int
        cost_hist,      ndarray (n_iter, 2)     [n_iter_in, cost]
        u,              ndarray (naux,)
        y,              ndarray (naux,)
    Note:
        ADMM with inequality constraint (if A provided):
            A * x >= 0
            
        Choose from following options for Primal subproblem:
            2nd-order method (Newton): Direct inversion (direct), Conjugate Gradient (CG) 
    """
    # If no constraint, set rho to 0 and A = np.eye(ndim)
    if A is None:
        A = np.eye(ndim)
        rho = 0
    
    naux = A.shape[0]

    # Initial guess
    #   Primal variable
    if x_init is None:
        x_init = np.zeros(ndim)
    y_init = np.zeros(naux)
    #   Dual variable
    u_init = np.zeros(naux)  

    # Outer loop for ADMM
    n_iter = 0
    upd = np.inf
    x, y, u = x_init, y_init, u_init
    cost_hist = np.empty((0,2))
    while n_iter < max_iter and upd > tol:
        
        # save old variable
        x_prev = x.copy()
        y_prev = y.copy()
        u_prev = u.copy()
        
        # ------------------ Primal subproblem 1 (Update x)--------------------- #
        # Newton update
        # calc. gradient
        g = func_g_cost(x) + rho * np.dot(A.T, np.dot(A, x) - y + u)
        # calc. hessian
        h = func_h_cost(x) + rho * np.dot(A.T, A)

        if solver == 'direct':
            dx, n_iter_in = np.dot(np.linalg.inv(h), g), 1
        elif solver == 'CG':
            # using CG
            dx, n_iter_in, *_ = CG(ndim, g, 
                                         lambda dx: np.dot(h, dx),
                                         **solver_param,
                                         verbose=False)
        x -= dx
        
        # ------------------ Primal subproblem 2 (Update y)--------------------- #
        # Soft-threshold
        y = np.maximum(MARGIN, np.dot(A, x) + u)
        
        # ------------------ Dual ascent (update u)----------------------- #
        u = u + np.dot(A, x) - y
        
        # calc. cost
        cost_x = func_cost(x)
        cost_hist = np.concatenate((cost_hist, np.array([[n_iter_in, cost_x]])), axis=0)
        
        # for stopping criterion
        n_iter += 1
        dx = x - x_prev
        upd = np.dot(dx.T, dx) / (np.dot(x.T, x) + EPS)
        
        # plot
        if verbose and n_iter%N_ITER_SHOW == 0:
            print('Iter {0}, (nested: {1}) Cost {2:.3f}, Upd (rel) {3:.3e}'.format(n_iter, n_iter_in, cost_x, upd))

    return x, n_iter, cost_hist, y, u