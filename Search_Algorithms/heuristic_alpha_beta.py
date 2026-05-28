from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple


def heuristic_alpha_beta_decision(
    state: Any,
    depth_limit: int,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    heuristic_fn: Callable[[Any], float],
) -> Tuple[Any, float]:
    """Alpha-beta search with a depth cutoff and a heuristic evaluation.

    When the cutoff is reached on a non-terminal state, `heuristic_fn(state)`
    is used instead of the exact terminal utility.
    """
    actions = list(actions_fn(state))
    if not actions:
        return None, utility_fn(state) if terminal_test(state) else heuristic_fn(state)

    best_action = None
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for action in actions:
        value = _min_value(
            result_fn(state, action),
            depth_limit - 1,
            actions_fn,
            result_fn,
            terminal_test,
            utility_fn,
            heuristic_fn,
            alpha,
            beta,
        )
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)

    return best_action, best_value


def _evaluate(
    state: Any,
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    heuristic_fn: Callable[[Any], float],
) -> float:
    return utility_fn(state) if terminal_test(state) else heuristic_fn(state)


def _max_value(
    state: Any,
    depth_left: int,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    heuristic_fn: Callable[[Any], float],
    alpha: float,
    beta: float,
) -> float:
    if depth_left == 0 or terminal_test(state):
        return _evaluate(state, terminal_test, utility_fn, heuristic_fn)

    actions = list(actions_fn(state))
    if not actions:
        return _evaluate(state, terminal_test, utility_fn, heuristic_fn)

    value = float("-inf")
    for action in actions:
        value = max(
            value,
            _min_value(
                result_fn(state, action),
                depth_left - 1,
                actions_fn,
                result_fn,
                terminal_test,
                utility_fn,
                heuristic_fn,
                alpha,
                beta,
            ),
        )
        if value >= beta:
            return value
        alpha = max(alpha, value)
    return value


def _min_value(
    state: Any,
    depth_left: int,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    heuristic_fn: Callable[[Any], float],
    alpha: float,
    beta: float,
) -> float:
    if depth_left == 0 or terminal_test(state):
        return _evaluate(state, terminal_test, utility_fn, heuristic_fn)

    actions = list(actions_fn(state))
    if not actions:
        return _evaluate(state, terminal_test, utility_fn, heuristic_fn)

    value = float("inf")
    for action in actions:
        value = min(
            value,
            _max_value(
                result_fn(state, action),
                depth_left - 1,
                actions_fn,
                result_fn,
                terminal_test,
                utility_fn,
                heuristic_fn,
                alpha,
                beta,
            ),
        )
        if value <= alpha:
            return value
        beta = min(beta, value)
    return value
