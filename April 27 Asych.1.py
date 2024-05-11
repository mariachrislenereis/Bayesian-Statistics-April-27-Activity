# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:05:51 2024

@author: mcreis
"""

import numpy as np

# Function to calculate the likelihood
def likelihood(mu, data):
    return np.prod(np.exp(- (data - mu)**2 / 2) / np.sqrt(2 * np.pi))

# Generate observed data
np.random.seed(123)
observed_data = np.random.normal(loc=5, scale=2, size=100)

# Initialize parameters
mu_current = 0  # initial guess for μ
sigma = 1       # known standard deviation
n_samples = 20000

# Store the samples
samples = []

for i in range(n_samples):
    # Propose new candidate from a symmetrical distribution
    mu_proposal = np.random.normal(loc=mu_current, scale=0.5)

    # Calculate likelihoods
    likelihood_current = likelihood(mu_current, observed_data)
    likelihood_proposal = likelihood(mu_proposal, observed_data)

    # Acceptance probability
    p_accept = min(1, likelihood_proposal / likelihood_current)

    # Accept proposal?
    accept = np.random.rand() < p_accept

    if accept:
        # Update position
        mu_current = mu_proposal

    samples.append(mu_current)
    
import pandas as pd
pd.DataFrame(samples).tail(100).hist()    
