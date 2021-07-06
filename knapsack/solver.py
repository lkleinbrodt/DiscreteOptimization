
#!/usr/bin/python
# -*- coding: utf-8 -*-
#%%
from collections import namedtuple
import numpy as np
import functools
import random

Item = namedtuple("Item", ['index', 'value', 'weight'])
recursion_counter = 0

def optimistic_estimate(items, capacity):
    sorted_items = sorted(items, key = lambda x: x.weight/x.value) #most valuable per weight is first
    cumulative_weights = np.cumsum([x.weight for x in sorted_items])
    over_capacity = cumulative_weights > capacity

    if cumulative_weights[0] > capacity:
        return (capacity / sorted_items[0].weight) * sorted_items[0].value
    elif sum(over_capacity) == 0:
        return evaluate_value(items)
    else:
        first_cant_fit = np.where(over_capacity)[0][0]
        fractional_space = capacity - cumulative_weights[first_cant_fit-1]
        fractional_value = fractional_space * (sorted_items[first_cant_fit].value / sorted_items[first_cant_fit].weight)
        return fractional_value + sum([x.value for x in sorted_items[:first_cant_fit]])

def evaluate_value(items):
    return sum([item.value for item in items])

def evaluate_weight(items):
    return sum([item.weight for item in items])

def greedy_recursion(in_napsack, items_left, capacity_remaining):
    
    if len(items_left) == 1:
        if items_left[0].weight <= capacity_remaining:
            in_napsack.append(items_left[0])
        return evaluate_value(in_napsack), in_napsack

    #Can we take the item?
    if items_left[0].weight > capacity_remaining:
        return greedy_recursion(in_napsack, items_left[1:], capacity_remaining)
    else:
        return greedy_recursion(in_napsack + [items_left[0]], items_left[1:], capacity_remaining - items_left[0].weight)
    #value, in_bag = depth_first([], items, capacity, 0)
    #taken = [1 if item in in_bag else 0 for item in items]

@functools.lru_cache(maxsize = None)
def lds(items, napsack, capacity, n, greedy_repeat = 7):
    global best_value
    global recursion_counter
    global best_value
    recursion_counter += 1

    current_value = evaluate_value(napsack)

    #Base case, you have one item left
    if len(items) == 1:
        if items[0].weight <= capacity:
            napsack = napsack + (items[0], )
            #napsack.append(items[0])
        return evaluate_value(napsack), napsack

    #if you have more than one item left, you will check if you have any deviations left to use
    # if you do, run LDS while making a mistake, and while not
    # return the better one

    if n > 0:
        #Heuristic = Check optimistic, then take item when possible

        #Follow Heuristic
        ###Can we take the item?
        if items[0].weight <= capacity:
            #Could it be better?
            optimistic = optimistic_estimate(items, capacity) + current_value
            if optimistic > best_value:
                obey_value, obey_items =  lds(items[1:], napsack + (items[0], ), capacity - items[0].weight, n)
            else:
                obey_value, obey_items = current_value, napsack
        else:
            optimistic = optimistic_estimate(items[1:], capacity) + current_value
            if optimistic > best_value:
                obey_value, obey_items =  lds(items[1:], napsack, capacity, n)
            else: 
                obey_value, obey_items = current_value, napsack

        ### Dont follow the heuristic
        optimistic = optimistic_estimate(items[1:], capacity) + current_value
        if optimistic > best_value:
            disobey_value, disobey_items = lds(items[1:], napsack, capacity, n-1)
        else:
            disobey_value, disobey_items = current_value, napsack

    else: 
        return greedy(items, napsack, capacity, greedy_repeat)

    if np.max([obey_value, disobey_value]) < best_value:
        return current_value, napsack
    else:
        if obey_value > disobey_value:
            best_value = obey_value
            return obey_value, obey_items
        else:
            best_value = disobey_value
            return disobey_value, disobey_items

@functools.lru_cache(maxsize = None)
def depth_recursion(items, napsack, capacity):
    global recursion_counter
    global best_value
    recursion_counter += 1
    # if recursion_counter % 50000 == 0:
    #     print(recursion_counter)
    current_value = evaluate_value(napsack)

    if len(items) == 1:
        if items[0].weight <= capacity:
            napsack = napsack + (items[0], )
        return evaluate_value(napsack), napsack
    
    #check yes value
    item_weight = items[0].weight
    if item_weight <= capacity:
        yes_optimistic = optimistic_estimate(items, capacity) + current_value
        if yes_optimistic > best_value:
            yes_value, yes_items = depth_recursion(items[1:], napsack + (items[0], ), capacity = capacity - item_weight) 
            #yes_value += current_value
            #print(f'Optimistic value of taking the item {yes_optimistic} is higher than best value {best_value}')
        else:
            yes_value, yes_items = current_value, napsack
            #print(f'Optimistic value of taking the item {yes_optimistic} is NOT higher than best value {best_value}')
    else:
        yes_value = current_value
        yes_items = napsack
    
    #check no value
    no_optimistic = optimistic_estimate(items[1:], capacity) + current_value
    if no_optimistic > best_value:
        #print(f'Optimistic no value: {no_optimistic} is greater than current best_value {best_value}')
        no_value, no_items = depth_recursion(items[1:], napsack, capacity = capacity)
        #no_value += current_value
    else:
        no_value, no_items = current_value, napsack

    best_value = np.max([best_value, yes_value, no_value, current_value])

    if yes_value > no_value:
        return yes_value, yes_items
    else:
        return no_value, no_items

def parse_input(file):
    with open(file, 'r') as input_data_file:
        input_data = input_data_file.read()
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    return items, capacity

items, capacity = parse_input('./data/ks_82_0')

def greedy(items, napsack, capacity, reps=7):
    initial_value = evaluate_value(napsack)
    initial_weight = evaluate_weight(napsack)

    items = sorted(items, key = lambda x: x.weight/x.value) #most valuable per weight is first

    def one_greedy(items, value, weight, napsack):
        for item in items:
            if weight + item.weight <= capacity:
                napsack = napsack + (item, )
                value += item.value
                weight += item.weight
        return (value, napsack)
    results = []
    for i in range(reps):
        if i == 0:
            results.append(one_greedy(items, initial_value, initial_weight, napsack))
        else:
            random.shuffle(items)
            results.append(one_greedy(items, initial_value, initial_weight, napsack))
    
    max_value = max([c[0] for c in results])
    return [(c[0], c[1]) for c in results if c[0] == max_value][0]

    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    global best_value
    best_value = 0 
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    items = tuple(items)
    if item_count < 200:
        value, napsack = depth_recursion(items, (), capacity)
    elif item_count == 200:
        value, napsack = lds(items, (), capacity, n = 4)
    elif item_count == 400:
        value, napsack = lds(items, (), capacity, n = 2)
    elif item_count == 1000:
        value, napsack = lds(items, (), capacity, n = 0, greedy_repeat = 10000)
    elif item_count == 10000:
        value, napsack = lds(items, (), capacity, n = 0, greedy_repeat = 5000)

    taken = [1 if item in napsack else 0 for item in items]
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))

    return output_data

#%%
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


# %%
