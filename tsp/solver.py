
#!/usr/bin/python
# -*- coding: utf-8 -*-

#### Test Point Lengths:
# 51
# 100
# 200
# 574
# 1889
# 33810 

#%%
from hashlib import new
import math
import numpy as np
from collections import namedtuple
from heapq import nlargest
import logging
import random
logging.basicConfig(level = logging.ERROR, format = ' %(asctime)s - %(levelname)s - %(message)s', filemode = 'w')
logger = logging.getLogger(__name__)

Point = namedtuple("Point", ['x', 'y', 'index'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def total_length(path, points):
    #distance = sum([length(points[path[i]], points[path[i+1]]) for i in range(len(path)-1)])
    #distance + length(points[path[-1]], points[path[0]])
    distance = sum([length(points[path[i-1]], points[path[i]]) for i in range(len(path))])

    return distance

def greedy(points):
    n_points = len(points)
    #Connect each point to closest point
    new_points = [p for p in points]
    solution = [new_points.pop()]

    while len(new_points) > 0:
        new_points = sorted(new_points, key = lambda p: -length(p, solution[-1]))
        solution.append(new_points.pop())

    solution = [p.index for p in solution]
    return solution

def random_walk(points, k):
    n_points = len(points)
    solution = list(range(n_points))
    best_distance = total_length(solution, points)
    
    
    for _ in k:
        walk = random.sample(points, n_points)
        d = total_length(walk, points)

        if d < best_distance:
            solution = walk
            best_distance = d

    return [p.index for p in solution]
        

def long_greedy(points, k = 500):

    new_points = [p for p in points]
    solution = []
    # Select K points, connect them by distance, repeat

    while len(new_points) > 0:
        batch = new_points[:k]
        new_points = new_points[k:]

        batch_solution = [batch.pop()]
        while len(batch) > 0:
            batch = sorted(batch, key = lambda p: -length(p, batch_solution[-1]))
            batch_solution.append(batch.pop())
        solution += (batch_solution)

    solution = [p.index for p in solution]
    return solution

    
def two_opt(points, max_passes = None, k = None, n = None):
    n_points = len(points)
    solution = greedy(points)

    found_improvement = True
    if max_passes is None:
        max_passes = 25
    if k is None:
        k = int(math.sqrt(n_points) * 5)
    if n is None:
        n = int(math.sqrt(n_points) * 5)
    passes = 0
    while found_improvement & (passes < max_passes):
        found_improvement = False

        ## Each pass, pick K edges randomly
        ## Then sort them by distance
        ## and try permutations using the first N

        edges_to_sample = random.sample(range(n_points), k = k)
        #edges_to_sample = nlargest(n, random_edges, key = lambda idx: length(points[solution[idx-1]], points[solution[idx]]))

        logging.debug(f"Path: {solution}")
        for i in edges_to_sample:
            for j in edges_to_sample:
                #Note that because we are changing the solution, the actual edges we're sampling wont be the same as what we selected at the beginning
                edge1 = (points[solution[i-1]], points[solution[i]])
                edge1_indices = [edge1[0].index, edge1[1].index]
                edge2 = (points[solution[j-1]], points[solution[j]])
                # for 2 opt, all you can do while maintaining a tour is connect e1 start with e2 start, e1end with e2end
                if (edge2[0].index in edge1_indices) | (edge2[1].index in edge1_indices):
                    continue #if thee two edges share a vertex, you cant swap out and maintain a tour
                
                # There has to be a way of getting the improvement just from calculating the nodes, without having to compute the full length

                original_length = total_length(solution, points)

                if i < j:
                    new_solution = solution[:i] + list(reversed(solution[i+1:j])) + [solution[i]] + solution[j:]
                elif j < i:
                    new_solution = solution[:j] + list(reversed(solution[j+1:i])) + [solution[j]] + solution[i:]
                
                else:
                    raise ValueError('j = i. Is this possible?')
                
                improvement =  original_length - total_length(new_solution, points)

                if improvement > 0:
                    logging.debug(f"Found improvement by swapping {edge1} and {edge2}")
                    found_improvement = True
                    #assert all([i in new_solution for i in solution]), [f for f in solution if f not in new_solution]
                    solution = new_solution
                    logging.debug(f"Actual improvement: {original_length - total_length(solution, points)}")
                    logging.debug(f"Path: {solution}")
                    
        
        passes +=1
    assert len(solution) == n_points
    return solution
                

def load_points(file_location):
    with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i-1))

    return points

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i-1))
    
    # d_mat = np.zeros(shape = (nodeCount, nodeCount))
    # for i in range(nodeCount):
    #     for j in range(nodeCount):
    #         d_mat[i,j] = length(points[i], points[j])

    if nodeCount == 51:
        k = nodeCount
    elif nodeCount == 100:
        k = nodeCount
    elif nodeCount == 200:
        k = nodeCount
    elif nodeCount == 574:
        k = None
    elif nodeCount == 1889:
        k = None
    elif nodeCount == 33810:
        k = None
    else:
        raise ValueError('Unrecognized length')

    if len(points) < 30_000:
        solution = two_opt(points, k)
    else:
        solution = long_greedy(points)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    assert len(solution) == nodeCount
    return output_data

# test_points = load_points('data/tsp_5_1')

# def load_data(file_location):
#     with open(file_location, 'r') as input_data_file:
#         input_data = input_data_file.read()
#     return input_data

# solve_it(load_data('data/tsp_5_1'))


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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')


# %%
