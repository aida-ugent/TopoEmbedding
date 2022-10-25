class NonAdaptiveSearcher:
    """
       Update algorithm that does not backtrack.
       Useful for loss function with sampling, that does
       not neccessarily decrease sufficiently.
    """

    def __init__(
        self,
        initial_step_size=1,
    ):
        self._initial_step_size = initial_step_size

    def search(self, objective, manifold, x, d, f0, df0):
        """Function to perform backtracking line search.
        Args:
            objective: Objective function to optimize.
            manifold: The manifold to optimize over.
            x: Starting point on the manifold.
            d: Tangent vector at ``x``, i.e., a descent direction.
            df0: Directional derivative at ``x`` along ``d``.
        Returns:
            A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
            the vector retracted to reach the suggested iterate ``newx`` from
            ``x``.
        """
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)

        alpha = self._initial_step_size / norm_d
        alpha = float(alpha)

        newx = manifold.retraction(x, alpha * d)
        step_size = alpha * norm_d
        return step_size, newx


class UnnormalizedSearcher:
    """
       Update algorithm that does not backtrack.
       Useful for loss function with sampling, that does
       not neccessarily decrease sufficiently.
    """

    def __init__(
        self,
        initial_step_size=1,
    ):
        self._initial_step_size = float(initial_step_size)

    def search(self, objective, manifold, x, d, f0, df0):
        """Function to perform backtracking line search.
        Args:
            objective: Objective function to optimize.
            manifold: The manifold to optimize over.
            x: Starting point on the manifold.
            d: Tangent vector at ``x``, i.e., a descent direction.
            df0: Directional derivative at ``x`` along ``d``.
        Returns:
            A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
            the vector retracted to reach the suggested iterate ``newx`` from
            ``x``.
        """
        # Compute the norm of the search direction
        newx = manifold.retraction(x, self._initial_step_size * d)
        return self._initial_step_size, newx
