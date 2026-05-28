
import networkx as nx

# Create directed graph
kg = nx.DiGraph()

# Add entities and relationships
kg.add_edge("Harsh", "Artificial Intelligence", relationship="studies")
kg.add_edge("Artificial Intelligence", "Computer Science", relationship="belongs_to")
kg.add_edge("Computer Science", "Technology", relationship="field_of")

print("\n🧠 Knowledge Graph Relationships\n")

# Display graph relationships
for source, target, data in kg.edges(data=True):
    print(f"{source} -- {data['relationship']} --> {target}")

print("\n✅ Knowledge Graph created successfully.")
