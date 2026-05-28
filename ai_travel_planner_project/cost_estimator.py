"""
Cost estimation module.

Produces a rough trip estimate using destination knowledge-base values,
travel duration, and a simple preference for comfort level.
"""

from __future__ import annotations

from typing import Dict

from knowledge_base import DESTINATIONS


COMFORT_MULTIPLIER: Dict[str, float] = {
    "budget": 0.9,
    "standard": 1.0,
    "premium": 1.35,
}


def estimate_trip_cost(destination: str, days: int, comfort: str = "standard") -> int:
    """
    Estimate total trip cost for a destination.

    Args:
        destination: Destination name from the knowledge base.
        days: Number of travel days.
        comfort: One of budget, standard, premium.

    Returns:
        Estimated total cost as an integer.
    """
    if destination not in DESTINATIONS:
        raise ValueError(f"Unknown destination: {destination}")
    if days <= 0:
        raise ValueError("days must be a positive integer")
    if comfort not in COMFORT_MULTIPLIER:
        raise ValueError("comfort must be one of: budget, standard, premium")

    base_daily = DESTINATIONS[destination].base_daily_cost
    multiplier = COMFORT_MULTIPLIER[comfort]
    transport_buffer = 1500
    total = int((base_daily * multiplier * days) + transport_buffer)
    return total


if __name__ == "__main__":
    print(estimate_trip_cost("Goa", 4, "standard"))
