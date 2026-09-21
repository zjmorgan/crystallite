"""Krylov solvers shared by the matrix-free spectral solvers."""

from crystallite.backend import xp


def preconditioned_conjugate_gradient(
    matvec, precondition, b, x0, tol=1.0e-6, max_iterations=500, divergence_factor=100.0
):
    r"""Solve :math:`A x = b` for a symmetric positive semi-definite `matvec`.

    Parameters
    ----------
    matvec : callable
        :math:`x \mapsto A x`.
    precondition : callable
        Applies an SPD approximate inverse of :math:`A` to a residual. Its
        choice changes the convergence rate, never the converged value.
    b : ndarray
        Right-hand side. Must be consistent (no component along :math:`A`'s
        null space).
    x0 : ndarray
        Initial guess.
    tol : float, default=1e-6
        Relative residual :math:`\|b - Ax\| / \|b\|` at which to stop.
    max_iterations : int, default=500
    divergence_factor : float or None, default=100
        Stop and return the best iterate once the residual exceeds this
        multiple of the best seen; ``None`` disables the guard. Only for
        systems with a near-null sector that round-off can excite; a
        legitimate (non-monotone) CG residual can blip by more than 100x
        at high contrast.

    Returns
    -------
    x : ndarray
        The best iterate seen (see the divergence guard in the source).
    residual_norm : float
        Relative residual of `x` (0 if ``b`` is numerically zero).
    iterations : int
    converged : bool
    """
    x = x0
    b_norm = float(xp.sqrt(xp.sum(b * b)))
    if b_norm < 1.0e-12:
        # x=0 already solves A x = 0; skip the relative residual, which
        # would divide by ~0.
        return x, 0.0, 0, True

    r = b - matvec(x)
    z = precondition(r)
    p = z
    rz_old = xp.sum(r * z)
    residual_norm = float(xp.sqrt(xp.sum(r * r))) / b_norm
    converged = residual_norm < tol
    iterations = 0
    # The divergence guard below may stop before the last step is the best.
    best_x, best_residual_norm = x, residual_norm

    while not converged and iterations < max_iterations:
        iterations += 1
        a_p = matvec(p)
        p_a_p = xp.sum(p * a_p)
        alpha = rz_old / p_a_p
        x = x + alpha * p
        r = r - alpha * a_p
        residual_norm = float(xp.sqrt(xp.sum(r * r))) / b_norm
        if residual_norm < tol:
            converged = True
            best_x, best_residual_norm = x, residual_norm
            break
        if residual_norm < best_residual_norm:
            best_x, best_residual_norm = x, residual_norm
        elif divergence_factor is not None and residual_norm > divergence_factor * best_residual_norm:
            # Divergence, not stalling: round-off can excite the near-null
            # eigenvalue sector of a pseudo-2D grid's out-of-plane DOF
            # (shape[2]==1), blowing up alpha with no recovery. Report the
            # best iterate seen instead of the runaway one. 100x, not a
            # tighter multiple: ordinary CG on a real heterogeneous problem
            # can blip to ~2.5x its best before converging normally; 2x cut
            # such a solve off early (a real regression this threshold
            # avoids).
            break
        z = precondition(r)
        rz_new = xp.sum(r * z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return best_x, best_residual_norm, iterations, converged
