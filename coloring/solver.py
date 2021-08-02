#!/usr/bin/python
# -*- coding: utf-8 -*-
#%%
from collections import namedtuple
import numpy as np
import functools
import random
import copy
import sys
import pandas as pd
import logging

### Node Stuff

class MyError(Exception):
    def __init__(self, message):
        self.message = message

class Node:
    def __init__(self, index, color = None, neighbors = []):
        self.index = index
        self.color = color
        self.neighbors = neighbors
        self.available_colors = None

    def get_available_colors(self, possible_colors):
        neighbor_colors = [n.color for n in self.neighbors]
        avail_colors = [c for c in possible_colors if c not in neighbor_colors]
        return avail_colors

    def prune_available_colors(self):
        if self.color is not None:
            self.available_colors = []
        elif len(self.available_colors) == 1:
            self.color = self.available_colors[0]
            self.available_colors = []
        else:
            self.available_colors = self.get_available_colors(self.available_colors)

    def append_available_color(self, new_color):
        if self.color is None:
            self.available_colors += [new_color]
    
    def set_degree(self):
        self.degree = len(self.neighbors)

def validate_node_list(node_list):
    return all([node.validate_color for node in node_list])

def get_max_degree(node_list):
    return max([len(n.neighbors) for n in node_list])

def get_min_degree(node_list):
    return min([len(n.neighbors) for n in node_list])

def get_min_colors(node_list):
    mutuals = [n.count_mutuals() for n in node_list]
    return max(mutuals) + 1

def copy_nodes(node_list):
    return [Node(n.index, n.color, n.neighbors) for n in node_list]

def print_colors(node_list):
    print([n.color for n in node_list])
    
def get_max_color(node_list):
    return max([n.color for n in node_list if n.color is not None] + [0])

### Greedy

def unbounded_greedy(node_list):
    """
    Doesnt ever look at "available colors", just starts with 0 and starts assigning
    """
    max_color = get_max_color(node_list)

    shuffled_list = random.sample(node_list, k = len(node_list))

    for node in shuffled_list:
        if node.color is not None:
            continue

        neighbor_colors = [n.color for n in node.neighbors if n.color is not None]
        if len(neighbor_colors) == 0:
            node.color = 0
        else:
            avail_colors= [c for c in range(max_color) if c not in neighbor_colors]

            if len(avail_colors) > 0:
                node.color = min(avail_colors)
            else:
                max_color += 1
                node.color = max_color

    return node_list

#def bounded_greedy(node_list):


def bounded_greedy_prune(node_list):
    """
    Assumes the nodes already have available colors listed, and tries to fill with those colors
    """
    [n.prune_available_colors() for n in node_list]
    #node_list = sorted(node_list, key = lambda x: -x.degree)

    next_uncolored = next((node for node in node_list if node.color is None), None)

    while next_uncolored is not None:

        try:    
            next_uncolored.color = next_uncolored.available_colors[0]   
            
        except IndexError:
            raise MyError('Unable to solve with this color pallete (bounded greedy prune')

        [node.prune_available_colors() for node in node_list if node.color is None]
        next_uncolored = next((node for node in node_list if node.color is None), None)

    return node_list 

def reset_nodes(node_list, max_color=None):
    for node in node_list:
        node.color = None
        if max_color is not None:
            node.available_colors = node.get_available_colors(possible_colors = list(range(max_color)))
    return node_list


def iterated_dropout(node_list, max_iterations):
    """
    
    """
    node_list = unbounded_greedy(node_list)
    upper_bound = get_max_color(node_list)+1
    print(f'Original Upper Bound: {upper_bound}')
    for node in node_list:
        node.color = None
        node.available_colors = list(range(upper_bound))

    node_list = bounded_greedy_prune(node_list)
    def one_iteration(node_list):

        #Pick 2 colors at random
        colors = list(range(get_max_color(node_list)))
        chosen_colors = random.sample(colors, 2)

        # We will reorder the nodes to go 1st group, 2nd group, rest (all in desc of degree)
        def node_val(node, group_one_color, group_two_color):
            #Smallest values go first
            if node.color == group_one_color:
                modifier = 10000
            elif node.color == group_two_color:
                modifier = 1000
            else:
                modifier = 0
            return modifier - node.degree

        node_list = sorted(node_list, key = lambda x: node_val(x, chosen_colors[0], chosen_colors[1]))

        #Now, erase those two colors
        for node in node_list:
            if node.color in chosen_colors:
                node.color = None
                node.available_colors = node.get_available_colors(possible_colors = colors)

        return bounded_greedy_prune(node_list)


    tries = 0
    #best_solution = copy_nodes(node_list)
    #best_n = n_colors(best_solution)

    while tries < max_iterations:
        try:
            node_list = one_iteration(node_list)
            max_color = get_max_color(node_list)
            # print(f'Success! {tries}')
            # print(f"Max Color: {max_color}")
            if max_color < 3:
                tries = max_iterations
        except MyError:
            print('Failure')
        tries +=1

    return node_list

def repeated_greedy_prune(node_list, n_iterations):
    node_list = sorted(node_list, key = lambda x: -x.degree)
    node_list = unbounded_greedy(node_list)
    solved_n = get_max_color(node_list)

    try:
        clean_node_list = reset_nodes(node_list, solved_n)
        node_list = bounded_greedy_prune(clean_node_list)
    except:
        pass   

    best_solution = copy_nodes(node_list)
    best_n = solved_n
    upper_bound = best_n

    ## This just redoes everything and tries again
    i = 0
    while (i < n_iterations/2) & (best_n > 2):
        for node in node_list:
            node.color = None
            node.available_colors = list(range(upper_bound))
        try:
            node_list = bounded_greedy_prune(node_list)
            solved_n = get_max_color(node_list)

            if solved_n < best_n:
                best_solution = copy_nodes(node_list)
                best_n = solved_n
                upper_bound = best_n
        except MyError:
            pass
        i += 1

    ### Only drop certain colors
    i = 0
    while (i < n_iterations/2) & (best_n > 2):
        colors = list(range(get_max_color(node_list)))
        if len(colors) < 3:
            n_to_choose = 1
        else:
            n_to_choose = random.randint(2,3)

        chosen_colors = random.sample(colors, n_to_choose)

        # Segregate and sort
        erased_lists = [[] for _ in range(n_to_choose)]
        leftovers = []
        for node in node_list:
            matched = False
            for idx, color in enumerate(chosen_colors):
                if node.color == color:
                    node.color = None
                    node.available_colors = node.get_available_colors(possible_colors = list(range(upper_bound)))
                    erased_lists[idx].append(node)
                    matched = True
                    break
            if not matched:
                node.available_colors = node.get_available_colors(possible_colors = list(range(upper_bound)))
                leftovers.append(node)
        erased_lists.append(leftovers)
        node_list = []
        for l in erased_lists:
            node_list += sorted(l, key = lambda x: -x.degree)

        try:
            node_list = bounded_greedy_prune(node_list)
            solved_n = get_max_color(node_list)
            if solved_n < best_n:
                best_solution = copy_nodes(node_list)
                best_n = solved_n
                upper_bound = best_n
        except MyError:
            pass
        
        i += 1

    return best_solution

def generate_node_list(lines):
    

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    nodes = [Node(i, None, []) for i in range(node_count)]

    nodes = [Node(i, None, []) for i in range(node_count)]

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        #edges.append((int(parts[0]), int(parts[1])))

        nodes[int(parts[0])].neighbors += [nodes[int(parts[1])]]
        nodes[int(parts[1])].neighbors += [nodes[int(parts[0])]]

    for node in nodes:
        node.set_degree()
    
    return nodes

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    lines = input_data.split('\n')
    # parse the input
    nodes = generate_node_list(lines)

    if len(nodes) < 250:
        tries = 5000
    elif len(nodes) < 1000:
        tries = 100
    else:
        tries = 10
    solved_nodes = repeated_greedy_prune(nodes, tries)

    solved_colors = [node.color for node in solved_nodes]
    
    # prepare the solution in the specified output format
    output_data = str(max(solved_colors)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solved_colors))

    return output_data

def load_nodes(file_location):
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()

    lines = input_data.split('\n')

    nodes = generate_node_list(lines)

    return nodes
#%%
import sys
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
