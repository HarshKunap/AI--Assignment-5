"""
Entry point for the AI-based travel planner.
"""

from __future__ import annotations

from personalized_planner import create_personalized_plan


def main() -> None:
    print("=== AI Based Travel Planner ===")
    preference = input("Enter travel preference (Beach/Hill Station/Historical/Nature/Cultural): ").strip()
    budget = input("Enter budget (Low/Medium/High): ").strip()
    season = input("Enter season (Summer/Monsoon/Winter): ").strip() or "Winter"
    food_interest = input("Enter food interest (optional): ").strip()
    days = int(input("Enter number of days: ").strip())
    comfort = input("Enter comfort level (budget/standard/premium): ").strip() or "standard"

    plan = create_personalized_plan(
        preference=preference,
        budget=budget,
        days=days,
        season=season,
        food_interest=food_interest,
        comfort=comfort,
    )

    print("\n=== Personalized Travel Plan ===")
    if "message" in plan:
        print(plan["message"])
        return

    print(f"Destination: {plan['destination']}")
    print(f"Type: {plan['type']}")
    print(f"Best Season: {plan['best_season']}")
    print(f"Rating: {plan['rating']}")
    print(f"Recommended Food: {', '.join(plan['recommended_food'])}")
    print(f"Top Activities: {', '.join(plan['top_activities'])}")
    print(f"Days: {plan['days']}")
    print(f"Estimated Cost: ₹{plan['estimated_cost']}")
    print(f"Other Suggestions: {', '.join(plan['recommendations'])}")


if __name__ == "__main__":
    main()
