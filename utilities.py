### This script holds some utility functions.

import numpy as np
from scipy.stats import randint

## funtion categorize_token
# takes a vector of token values x
# returns a vector of token classes
def categorize_tokens(x):
    # declare an empty vector to hold the translated values
    categories = np.array([])
    
    # categorize each token based on its value
    for i in range(len(x)):
        if x[i] < 0:
            categories = np.append(categories, 0)
        elif x[i] <= 3:
            categories = np.append(categories, 1)
        elif x[i] <= 7:
            categories = np.append(categories, 2)
        elif x[i] <= 11:
            categories = np.append(categories, 3)
        elif x[i] <= 15:
            categories = np.append(categories, 4)
        else:
            categories = np.append(categories, 5)
    
    return(categories)
    
## function expand_inventory
# takes a list of vectors x and integer players
# returns a zero-padded array of token categories in inventories
def expand_inventory(x, players):
    # declare an array of -1 placeholders with the appropriate dimensions
    inventory = np.full((players, 32), -1)
    
    # insert the values from x into inventory
    for i in range(len(x)):
        inventory[i, :len(x[i])] = categorize_tokens(x[i])
        
    return(inventory)
    
## function expand_vector
# takes a vector x
# returns a zero-padded vector
def expand_vector(x):
    # append zeros until the vector reaches max length
    while len(x) < 32:
        x = np.append(x, 0)
    
    return(x)
    
## function death_drop
# takes a player inventory
# returns an index indicating which token to drop
def death_drop(inventory):
    # create a vector of zeroes
    inventory_mask = np.zeros(len(inventory))

    # select a token at random
    inventory_mask[randint.rvs(0, len(inventory))] = 1

    return inventory_mask
