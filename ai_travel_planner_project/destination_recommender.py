"""
Destination recommendation module.

Uses reusable knowledge-base attributes to recommend destinations based on
user preferences like travel style, budget, cuisine interest, and season.
"""

from __future__ import annotations

from typing import List

from knowledge_base import DESTINATIONS, Destination


def _score_destination(destination: Destination, preference: str, budget: str, season: str, food_interest: str) -> int:
    """Compute a simple heuristic score for a destination."""
    score = 0
    preference = preference.lower().strip()
    budget = budget.lower().strip()
    season = season.lower().strip()
    food_interest = food_interest.lower().strip()

    if destination.place_type.lower() == preference:
        score += 4
    if destination.budget_level.lower() == budget:
        score += 3
    if destination.best_season.lower() == season:
        score += 2
    if any(food_interest in cuisine.lower() for cuisine in destination.cuisine):
        score += 1
    score += int(destination.rating)
    return score


def recommend_destinations(
    preference: str,
    budget: str,
    season: str = "Winter",
    food_interest: str = "",
    top_k: int = 3,
) -> List[str]:
    """
    Recommend destinations based on a preference profile.

    Returns:
        A list of destination names ordered by relevance.
    """
    scored = []
    for destination in DESTINATIONS.values():
        score = _score_destination(destination, preference, budget, season, food_interest)
        scored.append((score, destination.name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:top_k]]


if __name__ == "__main__":
    print(recommend_destinations("Beach", "Medium", "Winter", "Seafood"))
