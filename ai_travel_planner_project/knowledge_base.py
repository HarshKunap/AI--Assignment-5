"""
Knowledge base for an AI-based travel planner.

This module simulates reusable domain knowledge such as tourist places,
cuisine, activities, cost hints, and best seasons. The planner reuses this
knowledge to recommend destinations and generate a personalized trip plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Destination:
    name: str
    place_type: str
    budget_level: str
    cuisine: List[str]
    activities: List[str]
    best_season: str
    base_daily_cost: int
    rating: float


DESTINATIONS: Dict[str, Destination] = {
    "Goa": Destination(
        name="Goa",
        place_type="Beach",
        budget_level="Medium",
        cuisine=["Seafood", "Goan curry", "Bebinca"],
        activities=["Beach hopping", "Water sports", "Sunset cruise"],
        best_season="Winter",
        base_daily_cost=3200,
        rating=4.8,
    ),
    "Manali": Destination(
        name="Manali",
        place_type="Hill Station",
        budget_level="Low",
        cuisine=["Siddu", "Thukpa", "Himalayan tea"],
        activities=["Snow view", "Trekking", "Local market"],
        best_season="Summer",
        base_daily_cost=2200,
        rating=4.7,
    ),
    "Jaipur": Destination(
        name="Jaipur",
        place_type="Historical",
        budget_level="Medium",
        cuisine=["Dal Baati Churma", "Lassi", "Kachori"],
        activities=["Fort visits", "Shopping", "Heritage walk"],
        best_season="Winter",
        base_daily_cost=2800,
        rating=4.6,
    ),
    "Kerala": Destination(
        name="Kerala",
        place_type="Nature",
        budget_level="High",
        cuisine=["Appam", "Puttu", "Kerala seafood"],
        activities=["Backwaters", "Ayurveda", "Wildlife safari"],
        best_season="Monsoon",
        base_daily_cost=5000,
        rating=4.9,
    ),
    "Udaipur": Destination(
        name="Udaipur",
        place_type="Historical",
        budget_level="High",
        cuisine=["Rajasthani thali", "Daal Baati", "Mohan Maas"],
        activities=["Lake tour", "Palace visit", "Boat ride"],
        best_season="Winter",
        base_daily_cost=4500,
        rating=4.8,
    ),
    "Mysuru": Destination(
        name="Mysuru",
        place_type="Cultural",
        budget_level="Low",
        cuisine=["Mysore Pak", "Idli", "Dosa"],
        activities=["Palace visit", "Heritage sightseeing", "Market tour"],
        best_season="Winter",
        base_daily_cost=2100,
        rating=4.5,
    ),
}


def list_destination_names() -> List[str]:
    """Return all destination names from the knowledge base."""
    return list(DESTINATIONS.keys())
