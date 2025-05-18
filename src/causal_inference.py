# General required libraries
import pandas as pd
import sys
import os
import warnings

# Function needed for causal infenece by python dowhy 
from dowhy import CausalModel

# And libraries required for the alternative method
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

# And finally set the system path to the package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings('ignore')

file_path = os.path.join(package_root, "data", "lalonde.csv")
data = pd.read_csv(file_path)

# Define the Causal Model: Specify the treatment, outcome, and covariates:

model = CausalModel(
    data=data,
    treatment='treat',
    outcome='re78',
    common_causes=['educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']
)

# Identify the estimand (the quantity that we are going to estimate)
identified_estimand = model.identify_effect(proceed_when_unidentifiable = True)

# And then estimate the causal effect in terms of ATE (average treatment effect)
desired_effect = "ate"
method = "backdoor.linear_regression"

estimate = model.estimate_effect(identified_estimand,
                                 method_name = method,
                                target_units = desired_effect,
                                method_params = {"weighting_scheme" : "ips_weight"})
print(f"The average stimated effect of training on income is: ${estimate.value:.2f}")

# Finally, permute the treat to see if a close effect still can be seen. If so, the observed effect was random. 
refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="placebo_treatment_refuter")
print("\nRefutation test:")
print("-" * 50)
print(refute_results)
print("/nRefutation conclusion:")
print("-" * 50)
p_value = refute_results.refutation_result.get('p_value', None)
if p_value <= 0.05:
    print("The estimated ATE is statistically strong and can be refuted.")
else:
    print("The estimated ATE is statistically strong and can not be refuted.")
    print("Refuter p-value:", p_value)