"""Gradient descent algorithms.

"""
import numpy as np


def simple_descent(objective, init_guess, gradient, find_gamma=None, stop_condition=None):
    """Simple gradient descent algorithm.

    ..math::
        \mathbf{x}_{n+1} = \mathbf{x}_n - \gamma \nabla f(\mathbf{x}_n)
    where :math:`f(\mathbf{x}_{n+1}) \leq f(\mathbf{x}_{n})`

    Parameters
    ----------
    objective : function
        Objective of the descent.
        Input is the quantity being optimized.
        Output is a scalar.
    init_guess : np.ndarray
        Initial guess for the descent.
    gradient : function
        Gradient of objective.
        Input is the quantity being optimized.
        Output is a vector.
    find_gamma : function
        Function for guessing the gamma.
        Hyperparameter (gamma) for the step size.
    stop_condition : function
    """
    if find_gamma is None:
        def find_gamma(prev_gamma=0.1, success=True):
            if success:
                return prev_gamma
            else:
                return prev_gamma * 0.9

    if stop_condition is None:
        def stop_condition(gradient, step_size):
            return np.allclose(step_size, 0.0, atol=1e-3, rtol=0)

    new_x = init_guess
    new_objective = objective(init_guess)
    new_gradient = gradient(init_guess)
    step_size = find_gamma()

    while not stop_condition(new_gradient, step_size):
        # save previous info
        prev_x = new_x
        prev_gradient = new_gradient
        prev_objective = new_objective

        # attempt to find a new step
        attempt_x = prev_x - step_size * prev_gradient
        attempt_objective = objective(attempt_x)
        # if attempt is successful, proceed
        if attempt_objective < prev_objective:
            new_x = attempt_x
            new_objective = attempt_objective
            new_gradient = gradient(new_x)
            step_size = find_gamma(prev_gamma=step_size, success=True)
        # otherwise adjust stepsize
        else:
            new_x = prev_x
            new_objective = prev_objective
            new_gradient = prev_gradient
            step_size = find_gamma(prev_gamma=step_size, success=False)

    return new_x, new_objective, new_gradient
