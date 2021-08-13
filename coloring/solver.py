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
logging.basicConfig(level = logging.INFO, filename='log.log', format = ' %(asctime)s - %(levelname)s - %(message)s', filemode = 'w')
logger = logging.getLogger()
logger.debug('Start of program')

### Node Stuff

class MyError(Exception):
    def __init__(self, message):
        self.message = message

class Node:
    def __init__(self, index, color = None, neighbors = [], available_colors = []):
        self.index = index
        self.color = color
        self.neighbors = neighbors
        self.available_colors = available_colors
        self.degree = len(neighbors)
        
    def get_available_colors(self, possible_colors):
        neighbor_colors = [n.color for n in self.neighbors if n.color is not None]
        avail_colors = [c for c in possible_colors if c not in neighbor_colors]
        return avail_colors

    def prune_available_colors(self):
        if self.color is not None:
            self.available_colors = []
        # elif len(self.available_colors) == 1:
        #     self.color = self.available_colors[0]
        #     self.available_colors = []
        else:
            self.available_colors = self.get_available_colors(self.available_colors)
        return self

    def append_available_color(self, new_color):
        if self.color is None:
            self.available_colors += [new_color]
        return self
    
    def set_degree(self):
        self.degree = len(self.neighbors)
        return self

    def validate_color(self, require_color = True):
        if require_color:
            return all([neighbor.color != self.color for neighbor in self.neighbors] + [self.color is not None])
        else:
            return all([neighbor.color != self.color for neighbor in self.neighbors])

def prune_colors(node_list):
    for node in node_list:
        node = node.prune_available_colors()
        if len(node.available_colors) == 1:
            node.color = node.available_colors[0]
            node.available_colors = []
            for neighbor in node.neighbors:
                neighbor = neighbor.prune_available_colors()
    return node_list

def validate_node_list(node_list, require_color = True):
    return all([node.validate_color(require_color) for node in node_list])

def get_max_degree(node_list):
    return max([len(n.neighbors) for n in node_list])

def get_min_degree(node_list):
    return min([len(n.neighbors) for n in node_list])

def get_min_colors(node_list):
    mutuals = [n.count_mutuals() for n in node_list]
    return max(mutuals) + 1

def copy_nodes(node_list):
    nodes = [Node(index = n.index, color = n.color, neighbors=[], available_colors=n.available_colors) for n in node_list]
    for node in node_list:
        new_node = [n for n in nodes if n.index == node.index][0]
        new_node.neighbors = [n for n in nodes if n.index in [neighbor.index for neighbor in node.neighbors]]
        new_node = new_node.set_degree()
    return nodes

def print_colors(node_list):
    print([n.color for n in node_list])
    
def get_max_color(node_list):
    return max([n.color for n in node_list if n.color is not None] + [0])

def show_nodes(nodes):
    for node in nodes:
        if node.validate_color():
            flag = ''
        else:
            flag = '***'
        logger.debug(f"{flag}Node {node.index} ({node.color}-{node.available_colors}): {[(n.index, n.color) for n in node.neighbors]}")

### Greedy

def unbounded_greedy(node_list):
    """
    Doesnt ever look at "available colors", just starts with 0 and starts assigning
    """
    max_color = get_max_color(node_list)

    for node in node_list:
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
    #assert validate_node_list(node_list)
    return node_list

#def bounded_greedy(node_list):


def bounded_greedy_prune(node_list):
    """
    Assumes the nodes already have available colors listed, and tries to fill with those colors
    """
    node_list = copy_nodes(node_list)
    logger.debug('Starting bounded greedy prune')
    show_nodes(node_list)
    #node_list = sorted(node_list, key = lambda x: -x.degree)

    next_uncolored = next((node for node in node_list if node.color is None), None)

    while next_uncolored is not None:
        try:    
            next_uncolored.color = next_uncolored.available_colors[0]
            
            
            #if not validate_node_list(node_list, require_color=False):
            #    raise MyError('invalid solution, startover')
        except IndexError:
            #show_nodes(node_list)
            raise MyError('Unable to solve with this color pallete (bounded greedy prune')
        
        node_list = prune_colors(node_list)
        logger.debug('---')
        show_nodes(node_list)
        #node_list = [node.prune_available_colors() for node in node_list]
        next_uncolored = next((node for node in node_list if node.color is None), None)
    
    if not validate_node_list(node_list):
        logger.info('Invalid solution created')
        show_nodes(node_list)
        raise MyError('Created an invalid solution')
    return node_list 

def reset_nodes(node_list, max_color=None):
    for node in node_list:
        node.color = None
        if max_color is not None:
            node.available_colors = list(range(max_color))
    return node_list

def repeated_greedy_prune(node_list, n_iterations):
    logger.debug('Starting repeated greedy prune')
    #node_list = sorted(node_list, key = lambda x: -x.degree)
    node_list = unbounded_greedy(node_list)
    solved_n = get_max_color(node_list)
    best_solution = copy_nodes(node_list)
    best_n = solved_n
    logger.info(f"Brute force best: {best_n}")
    node_list = reset_nodes(node_list, solved_n)
    upper_bound = best_n

    ## This just redoes everything and tries again
    # logger.debug('Starting full dropouts')
    # i = 0
    # while (i < n_iterations/2) & (best_n > 2):
    #     node_list = reset_nodes(node_list, upper_bound)
    #     try:
    #         node_list = bounded_greedy_prune(node_list)
    #         solved_n = get_max_color(node_list)
    #         logger.info(f'Success {i}')
    #         if solved_n < best_n:
    #             best_solution = copy_nodes(node_list)
    #             best_n = solved_n
    #             upper_bound = best_n
    #             logger.info(f"Found a better solution on iteration {i}: {best_n}")
    #     except MyError:
    #         logger.debug('Failed')
    #     i += 1
    #     if (i % 500) == 0:
    #         logger.info(f'Full Dropout Iteration {i}')

    ### Only drop certain colors
    logger.debug('Starting partial dropouts')
    node_list = copy_nodes(best_solution)
    i = 0
    while (i < n_iterations) & (best_n > 2):
        colors = list(range(get_max_color(node_list)))
        if len(colors) < 3:
            n_to_choose = 1
        else:
            #n_to_choose = random.randint(2,3)
            n_to_choose = 2
        try:
            chosen_colors = random.sample(colors, n_to_choose)
        except ValueError as e:
            logger.info(f"Colors {colors}")
            raise e

        # Segregate and sort
        erased_lists = [[] for _ in range(len(chosen_colors))]
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
            node_list = reset_nodes(node_list, upper_bound)
            #node_list = bounded_greedy_prune(node_list)
            node_list = unbounded_greedy(node_list)
            solved_n = get_max_color(node_list)
            if solved_n < best_n:
                logger.info(f"Found a better solution on iteration {i}: {best_n}")
                best_solution = copy_nodes(node_list)
                best_n = solved_n
                upper_bound = best_n
        except MyError:
            node_list = copy_nodes(best_solution)
        
        i += 1
        # if (i % 500) == 0:
        #     logger.info(f'Partial Dropout Iteration {i}')

    return best_solution

def generate_node_list(lines):
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []

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

    if len(nodes) < 50:
        tries = 250
    elif len(nodes) == 50:
        tries = 500_000
    elif len(nodes) == 70:
        tries = 150_000
    elif len(nodes) == 100:
        tries = 100000
    elif len(nodes) == 250:
        tries = 4000
    elif len(nodes) == 500:
        tries = 400
    elif len(nodes) == 1000:
        tries = 4
    else:
        tries = 2
    
    logger.info(f'Starting optimization for {len(nodes)} nodes ({tries} iterations)')
    solved_nodes = repeated_greedy_prune(nodes, tries)
    logger.info(f'Finished optimization for {len(nodes)} nodes ({tries} iterations)')
    assert validate_node_list(solved_nodes)
    
    solved_nodes = sorted(solved_nodes, key = lambda x: x.index)
    solved_colors = [node.color for node in solved_nodes]
    
    # prepare the solution in the specified output format
    output_data = str(max(solved_colors)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solved_colors))
    
    with open(f"data/solutions/{len(nodes)}.txt", 'w') as f:
        f.write(output_data)
    show_nodes(solved_nodes)

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
