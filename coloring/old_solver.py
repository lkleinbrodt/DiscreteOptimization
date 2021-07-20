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
logging.basicConfig(level = logging.DEBUG,filename = 'log.txt', filemode = 'w', format=' %(asctime)s -  %(levelname)s -  %(message)s')
logging.debug('Start of program')

from numpy.core.defchararray import lower, upper

class MyError(Exception):
    def __init__(self, message):
        self.message = message

class Node:
    def __init__(self, index, color = None, neighbors = []):
        self.index = index
        self.color = color
        self.neighbors = neighbors
        self.available_colors = None
        self.degree = self.get_degree()

    def get_available_colors(self, possible_colors):
        neighbor_colors = [n.color for n in self.neighbors]
        avail_colors = [c for c in possible_colors if c not in neighbor_colors]
        return avail_colors

    def validate_color(self):
        no_nones = all([self.color is not None for c in self.neighbors])
        no_dupes = not any([self.color == c.color for c in self.neighbors])
        return no_nones & no_dupes

    def count_mutuals(self):
        mutuals = [sum([1 if self in n.neighbors else 0 for n in neighbor.neighbors]) for neighbor in self.neighbors]
        return int(np.floor(max(mutuals) / 2))

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

    def get_degree(self):
        return len(self.neighbors)


def iterated_greedy_prune(node_list, max_tries):
    node_list = greedy_prune(node_list)

    def one_iteration(node_list):
        #Pick 2 colors at random
        colors = range(max([n.color for n in node_list if n.color is not None]))
        chosen_colors = random.sample(colors, 2)
        #print(chosen_colors)
        
        # We will reorder the nodes to go 1st group, 2nd group, rest (all in desc of degree)
        def node_val(node, group_one_color, group_two_color):
            #Smallest values go first
            if node.color == group_one_color:
                modifier = -10000
            elif node.color == group_two_color:
                modifier = -1000
            else:
                modifier = 0
            return modifier - node.degree

        node_list = sorted(node_list, key = lambda x: node_val(x, chosen_colors[0], chosen_colors[1]))
        #print([n.index for n in node_list])
        #Now, erase those two colors
        for node in node_list:
            if node.color in chosen_colors:
                node.color = None

        return greedy_prune(node_list)

    #best_score = max([node.color for node in node_list])

    tries = 0
    while tries < max_tries:
        try:
            node_list = one_iteration(node_list)
            max_color = max([n.color for n in node_list])
            print(f'Success! {tries}')
            print(f"# of Colors: {max_color}")
        except MyError:
            print('Failure')
        tries +=1

    return node_list


def greedy_prune(node_list):

    counter = 0
    [n.prune_available_colors() for n in node_list]
    #node_list = sorted(node_list, key = lambda x: -x.degree)

    pick_top = True

    next_uncolored = next((node for node in node_list if node.color is None), None)

    while next_uncolored is not None:

        try:       
            # if pick_top:
            next_uncolored.color = next_uncolored.available_colors[0]
            # else:
            #     next_uncolored.color = next_uncolored.available_colors[1]
            # pick_top = not pick_top

            [node.prune_available_colors() for node in node_list if node.color is None]
        except IndexError:
            raise MyError('Unable to solve with this color pallete')

        # if counter % 10 == 0:
        #     print(f"Round: {counter}")
        #     #print(f"Colored Nodes: {n_colored}")
        #     print(f"Available Counts: ")
        #     print(pd.Series([len(n.available_colors) for n in node_list]).value_counts())

        # while not all([n.available_colors == n.get_available_colors(n.available_colors) for n in node_list]):
        #     [n.prune_available_colors() for n in node_list]

        next_uncolored = next((node for node in node_list if node.color is None), None)
        counter += 1

    return node_list 

def child_greedy_prune(node_list):

    n_colored = 0
    counter = 0
    [n.prune_available_colors() for n in node_list]

    while n_colored < len(node_list):

        counter +=1
        # print('New Round')
        # print('Colors')
        # print([n.color for n in node_list])
        # print('Available Colors')
        # print([n.available_colors for n in node_list])
        #random Node
        #chosen_node = random.sample([n.index for n in node_list if n.color is None], 1)[0]

        #Node with highest degree
        uncolored_nodes = [n for n in node_list if n.color is None]
        if len(uncolored_nodes) == 0:
            n_colored = len(node_list)
            continue
        else:
            #chosen_node = random.sample([n.index for n in node_list if n.color is None], 1)[0]
            chosen_node = [n.index for n in node_list if (len(n.neighbors) == get_max_degree(uncolored_nodes)) & (n.color is None)][0]
            try:       
                node_list[chosen_node].color = node_list[chosen_node].available_colors[0]
            except IndexError:
                # print(f"Chosen Node: {chosen_node}")
                # print(f"Node Available Colors: {node_list[chosen_node].available_colors}")
                # print(f"Neighbor Colors: {[no.color for no in node_list[chosen_node].neighbors]}")
                raise IndexError("Could not place color")
                # max_color = max([n.color for n in node_list])
                # node_list[chosen_node].color = max_color + 1
                # for node in uncolored_nodes:
                #     node.available_colors += max_color + 1
                
        n_colored = len(node_list) - len(uncolored_nodes) + 1 #cause we just colored one after counting the uncolored

        if counter % 10 == 0:
            print(f"Round: {counter}")
            print(f"Colored Nodes: {n_colored}")
            # print(f"Available Counts: ")
            # print(pd.Series([len(n.available_colors) for n in node_list]).value_counts())

        #while not all([n.available_colors == n.get_available_colors(n.available_colors) for n in node_list]):
        [n.prune_available_colors() for n in node_list]


    return node_list 

def checkpoint_prune(node_list):

    n_colored = 0
    counter = 0
    [n.prune_available_colors() for n in node_list]

    while n_colored < len(node_list):
        counter +=1
        # print('New Round')
        # print('Colors')
        # print([n.color for n in node_list])
        # print('Available Colors')
        # print([n.available_colors for n in node_list])
        #random Node
        #chosen_node = random.sample([n.index for n in node_list if n.color is None], 1)[0]

        #Node with highest degree
        uncolored_nodes = [n for n in node_list if n.color is None]
        if len(uncolored_nodes) == 0:
            continue
        else:
            #chosen_node = random.sample([n.index for n in node_list if n.color is None], 1)[0]
            chosen_node = [n.index for n in node_list if (len(n.neighbors) == get_max_degree(uncolored_nodes)) & (n.color is None)][0]

            success = False
            test_col = 0
            while not success:
                try:
                    tmp_list = copy.deepcopy(node_list)
                    tmp_list[chosen_node].color = tmp_list[chosen_node].available_colors[test_col]
                    print(f'Testing a greedy, {len(uncolored_nodes)} uncolored nodes left')

                    child_greedy_prune(tmp_list)
                    success = True
                    node_list[chosen_node].color = node_list[chosen_node].available_colors[test_col]

                except IndexError:
                    print('Trying another scenario')
                    test_col += 1

        n_colored = len([n for n in node_list if n.color is not None])

    return node_list 

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
    
def n_colors(node_list):
    return max([n.color for n in node_list])

def greedy(node_list, k = 1000):

    max_color = 0
    shuffled_list = random.sample(node_list, k = len(node_list))
    for node in shuffled_list:
        if node.color is not None:
            continue
        
        neighbor_colors = [n.color for n in node.neighbors if n.color is not None]
        if len(neighbor_colors) == 0:
            node.color = 0
        else:
            avail_colors = [c for c in range(max_color) if c not in neighbor_colors]
            if len(avail_colors) > 0:
                node.color = min(avail_colors)
            else:
                max_color += 1
                #print(f'Max: {max_color}')
                if max_color > k:
                    raise MyError(f'Unable to do with {k} colors')
                node.color = max_color

    return node_list

def repeated_greedy(node_list, max_tries):

    tmp_list = copy_nodes(node_list)
    lower_bound = get_min_colors(tmp_list)

    solved = False
    i = 0
    while (not solved) & (i < max_tries):
        try:
            output = greedy(copy_nodes(node_list), lower_bound)
            solved = True
            print(i)
        except MyError:
            i += 1
            output = None
    
    if output is not None:
        return output
    else:
        print('Nothing feasible, using greedy')
        return greedy(node_list)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    nodes = [Node(i, None, []) for i in range(node_count)]

    def generate_node_list():
        nodes = [Node(i, None, []) for i in range(node_count)]

        for i in range(1, edge_count + 1):
            line = lines[i]
            parts = line.split()
            #edges.append((int(parts[0]), int(parts[1])))

            nodes[int(parts[0])].neighbors += [nodes[int(parts[1])]]
            nodes[int(parts[1])].neighbors += [nodes[int(parts[0])]]
        return nodes

    nodes = generate_node_list()

    solved_nodes = greedy(nodes)
    upper_bound = max([n.color for n in solved_nodes]) + 1

    best_solution = copy_nodes(solved_nodes)
    best_n_colors = n_colors(best_solution)
    ## Keep trying to beat this score
    if len(nodes) == 250:
        pass
    elif len(nodes) == 1000:
        failed = False
        while not failed:
            try:
                nodes = generate_node_list()
                for node in nodes:
                    node.available_colors = list(range(upper_bound-1))
                solved_nodes = iterated_greedy_prune(nodes, max_tries=100)
                upper_bound -= 1
            except MyError:
                print(f'Failed to find solution with {upper_bound-1} colors')
                failed = True
    else:
        failed = False
        while (not failed) & (upper_bound > 1):
            try:
                print(f"Trying to find solution with {upper_bound} colors or less.")
                nodes = generate_node_list()
                for node in nodes:
                    node.available_colors = list(range(upper_bound-1))
                solved_nodes = iterated_greedy_prune(nodes, max_tries=5000)
                
                new_n_colors = n_colors(solved_nodes)
                print(new_n_colors)
                if  new_n_colors < best_n_colors:
                    print(f"New Best solution! {n_colors} beats {best_n_colors}")
                    best_solution = copy_nodes(solved_nodes)
                    best_n_colors = n_colors(best_solution)
                    upper_bound = best_n_colors
                upper_bound -= 1
            except MyError:
                print(f'Failed to find solution with {upper_bound} colors or less')
                failed = True

    # #did we find anything better than greedy?

    solved_colors = [node.color for node in best_solution]
    # prepare the solution in the specified output format
    output_data = str(max(solved_colors)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solved_colors))

    return output_data


import sys
#%%
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

#%%
file_location = 'data/gc_50_1'

with open(file_location, 'r') as input_data_file:
    input_data = input_data_file.read()

lines = input_data.split('\n')

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
nodes = [Node(i, None, []) for i in range(node_count)]

def generate_node_list():
    nodes = [Node(i, None, []) for i in range(node_count)]

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        #edges.append((int(parts[0]), int(parts[1])))

        nodes[int(parts[0])].neighbors += [nodes[int(parts[1])]]
        nodes[int(parts[1])].neighbors += [nodes[int(parts[0])]]
    return nodes

nodes = generate_node_list()

#%%
solve_it(input_data)
# %%
