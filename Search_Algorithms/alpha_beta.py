from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple


def alpha_beta_decision(
    state: Any,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
) -> Tuple[Any, float]:
    """Return the best action and its value using alpha-beta pruning."""
    actions = list(actions_fn(state))
    if not actions:
        return None, utility_fn(state)

    best_action = None
    alpha = float("-inf")
    beta = float("inf")
    best_value = float("-inf")

    for action in actions:
        value = _min_value(
            result_fn(state, action),
            actions_fn,
            result_fn,
            terminal_test,
            utility_fn,
            alpha,
            beta,
        )
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)

    return best_action, best_value


def _max_value(
    state: Any,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    alpha: float,
    beta: float,
) -> float:
    if terminal_test(state):
        return utility_fn(state)

    actions = list(actions_fn(state))
    if not actions:
        return utility_fn(state)

    value = float("-inf")
    for action in actions:
        value = max(
            value,
            _min_value(result_fn(state, action), actions_fn, result_fn, terminal_test, utility_fn, alpha, beta),
        )
        if value >= beta:
            return value
        alpha = max(alpha, value)
    return value


def _min_value(
    state: Any,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    utility_fn: Callable[[Any], float],
    alpha: float,
    beta: float,
) -> float:
    if terminal_test(state):
        return utility_fn(state)

    actions = list(actions_fn(state))
    if not actions:
        return utility_fn(state)

    value = float("inf")
    for action in actions:
        value = min(
            value,
            _max_value(result_fn(state, action), actions_fn, result_fn, terminal_test, utility_fn, alpha, beta),
        )
        if value <= alpha:
            return value
        beta = min(beta, value)
    return value
