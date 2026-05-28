import unittest
from dataclasses import dataclass, field
from typing import Dict, Optional

from minimax import minimax_decision
from alpha_beta import alpha_beta_decision
from heuristic_alpha_beta import heuristic_alpha_beta_decision
from mcts import mcts


@dataclass
class GameNode:
    name: str
    children: Dict[str, "GameNode"] = field(default_factory=dict)
    terminal_value: Optional[float] = None
    heuristic_value: float = 0.0

    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def actions(self):
        return list(self.children.keys())

    def result(self, action: str) -> "GameNode":
        return self.children[action]


def terminal_test(state: GameNode) -> bool:
    return state.is_terminal()


def utility_fn(state: GameNode) -> float:
    assert state.terminal_value is not None
    return state.terminal_value


def heuristic_fn(state: GameNode) -> float:
    return state.heuristic_value


class TestTreeSearchAlgorithms(unittest.TestCase):
    def setUp(self):
        # Exact minimax tree:
        # root(max)
        #   A -> min(3, 5) = 3
        #   B -> min(2, 4) = 2
        self.minimax_root = GameNode(
            "root",
            children={
                "A": GameNode("A", children={
                    "A1": GameNode("A1", terminal_value=3),
                    "A2": GameNode("A2", terminal_value=5),
                }),
                "B": GameNode("B", children={
                    "B1": GameNode("B1", terminal_value=2),
                    "B2": GameNode("B2", terminal_value=4),
                }),
            },
        )

        # Heuristic tree with depth 2:
        # At full depth, B is better (min(9,8)=8 > min(1,2)=1)
        # But at depth_limit=1, heuristic picks A because 10 > 0.
        self.heuristic_root = GameNode(
            "root",
            children={
                "A": GameNode(
                    "A",
                    heuristic_value=10,
                    children={
                        "A1": GameNode("A1", terminal_value=1),
                        "A2": GameNode("A2", terminal_value=2),
                    },
                ),
                "B": GameNode(
                    "B",
                    heuristic_value=0,
                    children={
                        "B1": GameNode("B1", terminal_value=9),
                        "B2": GameNode("B2", terminal_value=8),
                    },
                ),
            },
        )

        # Tiny MCTS tree: win is immediate; loss is immediate.
        self.mcts_root = GameNode(
            "root",
            children={
                "win": GameNode("win", terminal_value=1.0),
                "lose": GameNode("lose", terminal_value=0.0),
            },
        )

    def test_minimax(self):
        action, value = minimax_decision(
            self.minimax_root,
            lambda s: s.actions(),
            lambda s, a: s.result(a),
            terminal_test,
            utility_fn,
        )
        self.assertEqual(action, "A")
        self.assertEqual(value, 3)

    def test_alpha_beta(self):
        action, value = alpha_beta_decision(
            self.minimax_root,
            lambda s: s.actions(),
            lambda s, a: s.result(a),
            terminal_test,
            utility_fn,
        )
        self.assertEqual(action, "A")
        self.assertEqual(value, 3)

    def test_heuristic_alpha_beta(self):
        action, value = heuristic_alpha_beta_decision(
            self.heuristic_root,
            1,
            lambda s: s.actions(),
            lambda s, a: s.result(a),
            terminal_test,
            utility_fn,
            heuristic_fn,
        )
        self.assertEqual(action, "A")
        self.assertEqual(value, 10)

    def test_mcts(self):
        result = mcts(
            self.mcts_root,
            iterations=40,
            actions_fn=lambda s: s.actions(),
            result_fn=lambda s, a: s.result(a),
            terminal_test=terminal_test,
            reward_fn=utility_fn,
            seed=7,
        )
        self.assertEqual(result.action, "win")
        self.assertGreater(result.root_visits, 0)
        self.assertIn("win", result.action_stats)


if __name__ == "__main__":
    unittest.main()
