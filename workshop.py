import pandas as pd
import numpy as np
import networkx as nx
import math
from itertools import combinations
import os, platform, subprocess, re
import multiprocessing
import concurrent.futures




"""
As a bank robber, we want to ensure four things:
1) actually make it to the chopper on time
2) maximize the amount of money collected
3) prioritize the amount of time robbing a bank
   over the amount of time traveling between banks
4) point 3 because more banks robbed is likelier to maximize
   our profits than traveling a greater distance in 24hrs

To satisfy:
1) we should calculate FROM the endpoint (the origin)
2 - 4) 
    - we should affix an arbitrary value that represents
      the MAX amount of time (in hours) we're willing to travel between
      banks
    - this means using a graph and DF approach
    - this also means that to do this efficiently, while calculating
      distances between points, we should only consider adding edges to our graph
      that represent time travelled less than the MAX travel time allowed
    - this last point will weed out banks as possible next-stop destinations
      when at any given node.
    - given these conditions, this will also cut away banks that take too long
      to travel to the origin from, meaning, it will further weed out end-of-day
      banks, further reducing the number of potential robbery targets
"""

"""
Global variables that can be tweaked:
"""
time_threshold_for_edge_to_be_considered = 0.5
# The factor by which we multiply the mean of the values of time
# it takes to rob a bank in order to get the min threshold to 
# consider a bank as a node. The smaller this value, the more selective
# we are with our options
mean_time_factor=0.5
# The factor by which we multiply the mean of the values of bank vaults
# in order to get the min threshold to consider a bank as a node.
# The higher this value, the richer our banks need to be for us
# to consider them
mean_money_factor=2.9


"""
a and b: [tuple(x, y)]: cartesian coordinates of two points
"""
def distance_two_points(a, b, done=[]):
    if len(done) < len(a): 
        done.append(pow(a[len(done)] - b[len(done)], float(2)))
        return distance_two_points(a, b, done)
    inside = sum(done)
    return math.sqrt(inside)

"""
a and b: [tuple(x, y)]: cartesian coordinates of two points
maxT: [float]: max amount of time to spend traveling between a and b
vel: the velocity we'll be using
returns:
 - None if the distance should be ignored given maxT and vel
 - The amount of time (float > 0.0) should the value be lower than maxT
"""
def consider_distance_as_edge(a, b, maxT, vel):
    dist = distance_two_points(a, b)
    # print(dist)
    t = dist / vel
    return None if t > maxT else t


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

"""
Defaults to leaving 1 core thread free 
for laptops with more than 4 available 
threads
"""
def ideal_number_of_processes():
    count = multiprocessing.cpu_count()
    if count <= 2: return 2
    return count - 1


def split_list_for_multiprocess(lst, into=ideal_number_of_processes()):
    return list(chunks(lst, into))


def groups_to_split_origin_neighbors_into(size_batch):
    if size_batch < 50: 
        if size_batch <= ideal_number_of_processes() * 2: return 2
        return 1
    if 50 < size_batch < 150: return 2
    else: return 1

"""
MARK - GRAPH CLASSES
"""

class Bank:
    
    def __init__(self, _id, data=None):
        self._id = _id
        if data is not None:
            self.money = data.money
            self.time = data.time
        return

class Graph:

    completed = 0

    origin_neighors = set()

    def __init__(self):
        self.new_graph()
    
    def new_graph(self):
        self.G = nx.Graph([])
        self.G.add_node('o', weight=0.0, money=0)
        self.nodes = {'o': 0}
        self.origin_neighors.clear()

    def new_node(self, node, data=None):
        if self.G.has_node(node) is False:
            self.G.add_node(
                node,
                weight=(0.0 if data is None else data['time']),
                money=data['money']
            )
            # self.nodes[node] = data['money']

    def new_edge(self, A, B, t):
        # print(A, B)
        if self.G.has_edge(A, B) is False and self.G.has_edge(B, A) is False:
            # print(A, B, t)
            self.G.add_edge(A, B, weight=t)
        if str(A) == 'o' and str(B) != 'o':
            self.origin_neighors.add(B)
        if str(A) != 'o' and str(B) == 'o':
            self.origin_neighors.add(A)

    def longest_path(self):
        return nx.dag_longest_path(self.G)

    def neighbors_of_origin(self):
        return list(self.origin_neighors)

    def execute_sub_grouping(self, than, grouping, update_counts):
        results = []
        def run_edge(A, B, total=0,  amount=0, nodes=''):
            edge = self.G.get_edge_data(A,B)
            bWeight = self.G.nodes[B].get('node_weight', 1)
            next_total = edge['weight'] + bWeight + total
            next_amount = amount + self.G.nodes[B]['money']
            if next_total == total:
                results.append((nodes, amount))
                return
            elif next_total < than:
                neighbors = set()
                for node in self.G.neighbors(B):
                    if A != node and str(node) not in nodes:
                        neighbors.add(node)
                predecessors = set(nodes.split('-'))
                avail_successors = neighbors - predecessors
                if len(avail_successors) == 0:
                    results.append((nodes, amount))
                    return
                for successor in avail_successors:   
                    run_edge(
                        B,
                        successor,
                        total=next_total,
                        amount=next_amount,
                        nodes="{nodes}-{current}".format(nodes=nodes, current=B)
                    )
                return
            else:
                results.append((nodes, amount))
                return

        for item in grouping:
            run_edge('o', item, total=0.0, nodes="o")
            update_counts()
        return results


    def paths_with_time_less(self, than, neighs_arg=None):
        """
        This is our bottleneck
        """
        neighs = neighs_arg if neighs_arg is not None else self.neighbors_of_origin()
        results_sublists = []
        print(len(neighs), "neighbors to origin")
        print("The exit helicopter zone can be reached from {neig} neighboring banks".format(neig=len(neighs)))
        divided_neighbors = split_list_for_multiprocess(
            neighs,
            groups_to_split_origin_neighbors_into(len(neighs))
        )
        print("Spliting up the neighbors into groups of {spli}".format(spli=groups_to_split_origin_neighbors_into(len(neighs))))
        print("These are the divided neighbors to origin:")
        print(divided_neighbors)
        print(" ")
        print("Traversing our graph accross {cpus} processes....".format(cpus=ideal_number_of_processes()))

        def update_counts():
            self.completed += 1
            print("Done {comp}/{tots}".format(comp=self.completed,tots=len(neighs)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=ideal_number_of_processes()) as executor:
            paths_from = {executor.submit(
                self.execute_sub_grouping, 
                than, 
                grouping,
                update_counts
            ): grouping for grouping in divided_neighbors}
            for paths in concurrent.futures.as_completed(paths_from):
                paths_results = paths_from[paths]
                try:
                    data = paths.result()
                    results_sublists.append(data)
                except Exception as exc:
                    print('%r generated an exception: %s' % (paths_results, exc))

        print("Flattening the results")
        results = []
        for sublist in results_sublists:
            for result in sublist:
                results.append(result)
        return results


"""
Calculate acceptable distances from the origin
"""
def determine_edges_to_origin(df):
    def lambda_origin(row):
        edge = consider_distance_as_edge(
            (row.x_coordinate, row.y_coordinate), 
            (0,0), 
            time_threshold_for_edge_to_be_considered, 
            30
        )
        return -1 if edge is None else edge
    df['or'] = df.apply(lambda_origin, axis=1)

def filter_out_banks_with_means(df, mean_money_f=1.0, mean_time_f=0.5):
    mean = df.mean()
    dfc = df.copy()
    mean_money = mean['money'] * mean_money_factor
    mean_time = mean['time (hr)'] * mean_time_factor
    print(" ")
    print("We are NOW filtering banks who have at least ${money}".format(money=mean_money))
    print(" ")
    print("And we are filtering banks who take less than {time} hrs to rob".format(time=mean_time))
    print(" ")
    dfc_s = np.where((dfc['money'] >= mean_money) & (dfc['time (hr)'] <= mean_time))
    to_use = dfc[dfc['id'].isin(dfc_s[0])]
    return to_use

"""
Takes in the filtered dataframe, iterates over the bank id's,
then operates each iteration on the original dataframe from the .csv
file.
options: [Dataframe]: filtered
df: [Dataframe]: original
graph: [Graph]: the graph class we'll be using
"""
def iterate_and_apply_valid_edges(options, df, graph):
    # First filter out origin-distances that are not -1
    df_id1 = options['id'].to_list()
    counter = 0
    considered = 0
    combs = []
    for (A, B) in combinations(df_id1, 2):
        combs.append((A,B))
        if (counter % 10000 == 0):
            print('.')
        counter += 1
        edge = consider_distance_as_edge(
            (df.iloc[A, 1], df.iloc[A, 2]), 
            (df.iloc[B, 1], df.iloc[B, 2]), 
            0.3, 
            30
        )
        if edge is not None:
            # Add edges and nodes only if the nodes 
            # Make up an acceptable distance 
            # between them
            considered += 1
            bankA = {
                "money": df.iloc[A, 3],
                "time": df.iloc[A, 4],
            }
            bankB = {
                "money": df.iloc[B, 3],
                "time": df.iloc[B,4],
            }
            graph.new_node(A, bankA)
            graph.new_node(B, bankB)
            graph.new_edge('o', A, df.iloc[A, 5])
            graph.new_edge('o', B, df.iloc[B, 5])
            graph.new_edge(A, B, edge) 
    print("Done iterating over {count} bank-to-bank combinations".format(count=counter))
    print("Considered over {considered} bank-to-bank combinations".format(considered=considered))
    # print(combs)
    return

"""
Following methods returns the path that robbed the most money
"""
def sort_paths_by_sum_robbed(paths):
    paths.sort(key=lambda  x: x[1], reverse=True)

def return_best_path_of_banks(paths):
    sort_paths_by_sum_robbed(paths)
    first = paths[0][0].split('-')[1:]
    # We rebuild the array of integers from the array of strings
    return [int(i) for i in first][::-1]

"""
df: [string]: path to the file
"""
def robber_algorithm(path):

    df = pd.read_csv(path)
    determine_edges_to_origin(df)
    graph = Graph()

    def optimize_factors():
        to_use = filter_out_banks_with_means(
            df
        )
        iterate_and_apply_valid_edges(to_use, df, graph)
        if len(graph.neighbors_of_origin()) < 18: return
        else:
            print("Optimizing our selection criteria to narrow our options..")
            print(" ")
            print(" ")
            print(" ")
            print(" ")
            global mean_time_factor
            global mean_money_factor
            mean_time_factor -= 0.025
            mean_money_factor += 0.05
            graph.new_graph()
            return optimize_factors()

    optimize_factors()

    paths = graph.paths_with_time_less(24.0)
    result = return_best_path_of_banks(paths)
    print("The ideal path would be :")
    print(result)


robber_algorithm('bank_data.csv')
# print(path)
