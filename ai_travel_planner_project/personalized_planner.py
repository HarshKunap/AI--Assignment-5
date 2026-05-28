"""
Personalized tour planner.

Combines destination recommendation and cost estimation to generate a
simple trip plan.
"""

from __future__ import annotations

from typing import Dict, List

from knowledge_base import DESTINATIONS
from cost_estimator import estimate_trip_cost
from destination_recommender import recommend_destinations


def create_personalized_plan(
    preference: str,
    budget: str,
    days: int,
    season: str = "Winter",
    food_interest: str = "",
    comfort: str = "standard",
) -> Dict[str, object]:
    """
    Create a personalized travel plan from the knowledge base.
    """
    suggestions = recommend_destinations(
        preference=preference,
        budget=budget,
        season=season,
        food_interest=food_interest,
        top_k=3,
    )
    if not suggestions:
        return {
            "recommendations": [],
            "message": "No matching destinations found.",
        }

    chosen = suggestions[0]
    d = DESTINATIONS[chosen]

    plan = {
        "destination": d.name,
        "type": d.place_type,
        "recommended_food": d.cuisine[:3],
        "top_activities": d.activities[:3],
        "best_season": d.best_season,
        "rating": d.rating,
        "estimated_cost": estimate_trip_cost(d.name, days, comfort),
        "days": days,
        "recommendations": suggestions,
    }
    return plan


if __name__ == "__main__":
    from pprint import pprint
    pprint(create_personalized_plan("Beach", "Medium", 3, "Winter", "Seafood"))
