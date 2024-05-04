# Bayesian-Statistics-April-27-Activity
Research about PYMC3

# PyMC3 in Python

PyMC3 is a versatile programming library for probabilistic programming and Bayesian in Python. It comes up with a framework that can be used to create and fit Bayesian models. It enables to carry out Bayesian data analysis and produce probabilistic inferences. It is a useful tool for conducting Bayesian data analysis. Because of its efficient inference algorithms and model comparison tools.

Moreover, PyMC3 allows to use of a simple syntax for expressing complex models. Wherein, mathematical expressions can be used to create probabilistic variables, establish their prior distributions, and construct relationships between variables. PyMC3 runs Markov Chain Monte Carlo (MCMC) sampling in order to determine the posterior distribution of the model parameters. It offers several sampling algorithms, such as Metropolis-Hastings and the No-U-Turn Sampler (NUTS).

Additionally, PyMC3 provides tools for model comparison and selection which enables to compare various models and choose the one that most closely matches the data. It supports evaluating the model's fit and complexity with methods such as the Deviance Information Criterion (DIC) and the Widely Applicable Information Criterion (WAIC). Furthermore, through the model comparison and selection features of PyMC3 help to evaluate the suitability of several models and determine the best fit for the data.

*Note: Unable to import PyMC3. Thus, these are the provided samples.*

## Sample Script/Code

```python
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
```
## Screenshot of the Output

![APRIL 27 1](https://github.com/mariachrislenereis/Bayesian-Statistics-Activity-3/assets/168893458/d225f54a-4de3-4665-9fe4-2b0a05d6df05)

## Bayesian Model for Forecasting/Regression

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

# sample x
np.random.seed(2022)
x = np.random.rand(100)*30

# set parameters
a = 3
b = 20
sigma = 5


# obtain response and add noise
y = a*x+b
noise = np.random.randn(100)*sigma

# create a matrix containing the predictor in the first column
# and the response in the second
data = np.vstack((x,y)).T + noise.reshape(-1,1)

# plot data 
plt.scatter(data[:,0], data[:,1])
plt.xlabel("x")
plt.ylabel("y")
```

## Screenshot of the Output

![APRIL 27 2](https://github.com/mariachrislenereis/Bayesian-Statistics-Activity-3/assets/168893458/013af7d0-1496-449c-a1b0-0b916ec211e5)


## References

Aswani, H. (2023, October 31). Practical Applications of PyMC3 in Data Science. Medium. https://medium.com/@harshitaaswani2002/practical-applications-of-pymc3-in-data-science-85c967da79ad
Copley, C. (2023, June 16). Understanding the Monte Carlo Markov Chain: A Key to Bayesian Inference. Medium. https://charlescopley.medium.com/understanding-the-monte-carlo-markov-chain-a-key-to-bayesian-inference-163b03f9fd2d
Nucera, F. (2023, July 4). Mastering Bayesian Linear Regression from Scratch: A Metropolis-Hastings Implementation in Python. Medium. https://medium.com/@tinonucera/bayesian-linear-regression-from-scratch-a-metropolis-hastings-implementation-63526857f191
