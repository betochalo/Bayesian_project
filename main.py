import covariance_function as cf
import numpy as np 
import scipy


nb_of_samples = 41
number_of_functions = 5

X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)

print(X)