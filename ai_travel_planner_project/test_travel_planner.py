"""
Test cases for the AI-based travel planner.
"""

from __future__ import annotations

import unittest

from cost_estimator import estimate_trip_cost
from destination_recommender import recommend_destinations
from personalized_planner import create_personalized_plan


class TestTravelPlanner(unittest.TestCase):
    def test_recommend_destinations_returns_goa(self):
        result = recommend_destinations("Beach", "Medium", "Winter", "Seafood")
        self.assertIn("Goa", result)

    def test_cost_estimation_positive_integer(self):
        cost = estimate_trip_cost("Goa", 3, "standard")
        self.assertIsInstance(cost, int)
        self.assertGreater(cost, 0)

    def test_plan_contains_key_fields(self):
        plan = create_personalized_plan("Historical", "Medium", 2, "Winter", "Lassi")
        self.assertIn("destination", plan)
        self.assertIn("estimated_cost", plan)
        self.assertIn("recommendations", plan)

    def test_unknown_destination_raises(self):
        with self.assertRaises(ValueError):
            estimate_trip_cost("UnknownPlace", 2, "standard")


if __name__ == "__main__":
    unittest.main()
