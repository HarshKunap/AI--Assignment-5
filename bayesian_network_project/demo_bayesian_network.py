
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Rain', 'Sprinkler'),
    ('Rain', 'GrassWet'),
    ('Sprinkler', 'GrassWet')
])

# Define CPDs

cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[[0.7], [0.3]]
)

cpd_sprinkler = TabularCPD(
    variable='Sprinkler',
    variable_card=2,
    values=[
        [0.4, 0.9],
        [0.6, 0.1]
    ],
    evidence=['Rain'],
    evidence_card=[2]
)

cpd_grasswet = TabularCPD(
    variable='GrassWet',
    variable_card=2,
    values=[
        [0.99, 0.2, 0.1, 0.01],
        [0.01, 0.8, 0.9, 0.99]
    ],
    evidence=['Rain', 'Sprinkler'],
    evidence_card=[2, 2]
)

# Add CPDs
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_grasswet)

# Validate model
print("\n✅ Bayesian Network Valid:", model.check_model())

# Perform inference
inference = VariableElimination(model)

print("\n🧠 Probability of GrassWet given Rain=True\n")

result = inference.query(
    variables=['GrassWet'],
    evidence={'Rain': 1}
)

print(result)
