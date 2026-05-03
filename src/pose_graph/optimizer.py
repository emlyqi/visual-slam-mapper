"""
GTSAM Levenberg-Marquardt optimization wrapper for pose graphs.
"""

import gtsam


def optimize(graph, initial, max_iterations=100, verbose=True):
    """
    Run Levenberg-Marquardt optimization on a pose graph.
    Args:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values with initial pose estimates
        max_iterations: hard cap on iterations
        verbose: print convergence info
    Returns:
        result: gtsam.Values with optimized poses
        info: dict with diagnostic info (initial_error, final_error, iterations)
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    if verbose:
        params.setVerbosityLM("TERMINATION") # print convergence info at termination

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    initial_error = graph.error(initial)
    result = optimizer.optimize()
    final_error = graph.error(result)

    info = {
        'initial_error': float(initial_error),
        'final_error': float(final_error),
        'iterations': int(optimizer.iterations()),
        'error_reduction': float(initial_error - final_error),
        'error_reduction_pct': 100 * (initial_error - final_error) / max(initial_error, 1e-12),
    }

    if verbose:
        print(f"Initial error: {initial_error:.2f}")
        print(f"Final error:   {final_error:.2f}")
        print(f"Iterations:    {info['iterations']}")
        print(f"Reduction:     {info['error_reduction_pct']:.1f}%")

    return result, info