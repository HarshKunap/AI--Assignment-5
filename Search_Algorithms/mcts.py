from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class MCTSNode:
    state: Any
    parent: Optional["MCTSNode"] = None
    action_from_parent: Any = None
    children: Dict[Any, "MCTSNode"] = field(default_factory=dict)
    untried_actions: List[Any] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


@dataclass(frozen=True)
class MCTSResult:
    action: Any
    root_visits: int
    action_stats: Dict[Any, Tuple[int, float]]


def mcts(
    root_state: Any,
    iterations: int,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    reward_fn: Callable[[Any], float],
    exploration_constant: float = 1.41421356237,
    seed: Optional[int] = None,
) -> MCTSResult:
    """Monte-Carlo Tree Search using the UCT selection rule.

    `reward_fn(state)` must return the terminal reward from the root player's
    perspective. For a two-player zero-sum game, a common convention is:
    win = 1.0, draw = 0.5, loss = 0.0.
    """
    rng = Random(seed)
    root = MCTSNode(
        state=root_state,
        untried_actions=list(actions_fn(root_state)),
    )

    if terminal_test(root_state):
        return MCTSResult(action=None, root_visits=0, action_stats={})

    for _ in range(iterations):
        node = root
        state = root_state

        # Selection
        while node.children and node.fully_expanded() and not terminal_test(state):
            node = _uct_select(node, exploration_constant)
            state = node.state

        # Expansion
        if not terminal_test(state) and node.untried_actions:
            action = rng.choice(node.untried_actions)
            node.untried_actions.remove(action)
            next_state = result_fn(state, action)
            child = MCTSNode(
                state=next_state,
                parent=node,
                action_from_parent=action,
                untried_actions=list(actions_fn(next_state)),
            )
            node.children[action] = child
            node = child
            state = next_state

        # Simulation
        reward = _rollout(
            state,
            actions_fn,
            result_fn,
            terminal_test,
            reward_fn,
            rng,
        )

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    best_action = None
    best_visits = -1
    stats: Dict[Any, Tuple[int, float]] = {}
    for action, child in root.children.items():
        stats[action] = (child.visits, child.mean_reward)
        if child.visits > best_visits:
            best_visits = child.visits
            best_action = action

    return MCTSResult(action=best_action, root_visits=root.visits, action_stats=stats)


def _uct_select(node: MCTSNode, exploration_constant: float) -> MCTSNode:
    assert node.children, "UCT selection requires at least one child."
    log_parent = log(node.visits) if node.visits > 0 else 0.0

    def uct_score(child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.mean_reward
        exploration = exploration_constant * sqrt(log_parent / child.visits) if child.visits > 0 else float("inf")
        return exploitation + exploration

    return max(node.children.values(), key=uct_score)


def _rollout(
    state: Any,
    actions_fn: Callable[[Any], Iterable[Any]],
    result_fn: Callable[[Any, Any], Any],
    terminal_test: Callable[[Any], bool],
    reward_fn: Callable[[Any], float],
    rng: Random,
) -> float:
    current = state
    while not terminal_test(current):
        actions = list(actions_fn(current))
        if not actions:
            break
        action = rng.choice(actions)
        current = result_fn(current, action)
    return reward_fn(current)
