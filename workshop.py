import pandas as pd
import numpy as np
import networkx as nx
import math
from itertools import combinations
import os, platform, subprocess, re
import multiprocessing
import concurrent.futures
from check_solution import check_solution



"""
As a bank robber, we want to ensure four things:
1) actually make it to the chopper on time
2) maximize the amount of money collected
3) prioritize the amount of time robbing a bank
   over the amount of time traveling between banks
4) point 3 because more banks robbed is likelier to maximize
   our profits than traveling a greater distance in 24hrs

Algorithmically we want to ensure the following:
 - that the algorithm returns a value within 3 minutes
 - that a value is returned by assuming the device computing
   the algorithm is a commercial laptop (so 4-6 core CPU),
   with no dedicated GPU.

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


DESCRIPTION OF ASSUMPTIONS AND CONVENTIONS TO TAKE

- Since we are operating under the assumption that the device is a 4-6 core CPU,
  we can assume it will likely have 12 threads at most. 
- so as to not freeze the device on the machine, we will make sure 2 threads remain free
  for other tasks running on the device
- this means we need to consider how many possible paths we can evaluate for time and money
  given that we might have a low thread count to utilize
- the simplest approach to solving this logistical issue is to take an Optimization approach
- we will optimize our list of possible banks according to three consideration variableS:
    A: how long we're willing to spend travelling from bank to bank (weight of an edge)
    B: how much time we're willing to spend at a bank to rob it
    C: the minimum amount a bank must have in order for us to consider robbing it 
- what we will do is iterate over smaller and smaller filtered out results of the list of banks
  with these three factors being our filtering variables
- at each iteration, we will check how many banks remain
- Since we are operating under the assumption that we do not have a speed-efficient CPU on hand:
    - we will set an arbitrary value of maximum banks we need to filter down to before running our analysis
    - once all possible routes are mapped out by total time spend and sum of money robbed, we
      will sort out the route with the highest amount robbed (since this is our main goal as robbers)
- A corolary of our conditions above means that our optimized list will have only banks whose time to travel
    to from the origin is less than our maximum time traveled variable. Since these banks represent final banks robbed in any banks
    this further accelerates the rate at which we reach the pool of optimized banks.
- Regarding how we will keep track of paths taken:
    - we're looking at remembering order of the path taken, essentially, a chain of past banks visited from
      the point of view of any bank we're currently at in a path's analysis
    - what we cannot do is use a set to transmit what we've visited in the past to the next bank we're iterating to
    - we will use a list-compatible data format, such as a character-seperated string sequence (A-B-C-D-E), or a list
    - to avoid any method arguments conflict with lists, we'll pass along a character-seperated string sequence, while
      exctracting its chained banks into a set at each iteration to make sure we don't re-iterate over a bank we've already
      visited


DESCRIPTION OF MAJOR STEPS:

- first, we will optimize our list of banks to consider based on the points above
- second, we will build our graph with the optimized list of banks
- we will perform depht-first searches for all paths from the extraction point (the origin) that take less than
  the alocated time to rob banks, by passing through the origin's neighbors (via networkX)
  - third, in order to speed things up, we will apply a divide-and-conquer approach over paths that go through 
    the neighbors. 
  - by considering how many threads are available to us, we will apply a small algorithm to determine
    how many neighbors of the origin to pass to each thread. Starting with one each, we will assign as many to each thread
    while trying to underwork the smallest number of threads possible.
  - Once all processes have completed, we return the unsorted list
- Fourth, we sort the list in descending order by money stolen, and extract the list's first element
- Fifth, since the path contained in the most lucrative path returned is a string (A-B-C-D...)
    - we split the string into a list of bank id's
    - then since this list begins from the origin's neighbor, and since we want a list that begins at the bank we
      should start at, we return the reverse of this list of bank id's.


VARIABLES:

A: time_threshold_for_edge_to_be_considered, float, representing the amount of hours we're willing to spend
    travelling between banks, this value is fixed but can be changed by the user.
B and C: since we're dealing with a considerable amount of banks to consider, the simplest approach is to
    consider the:
    -  B: "above average" banks in terms of worth, since we can be sure that robbing any number X banks that have
        a worth above the average worth of all banks, will net us more money than robbing any number X banks that have
        a less than average worth, regardless of time. Should we consider bellow-average banks, this will mean that we
        would need to spend MORE time robbing, which could turn out not to be more time-efficient, or could perhaps
        even increase the chance of us getting robbed.
    -  C: "bellow average"  banks in terms of time spent robbing them, since while we want to rob as many banks as possible,
        we also do not want to rob alot of banks needlessly or blindly. We want to get the biggest bang for our time spent,
        so considering banks who take too long for our liking may not prove to be beneficial in the end.

D: max number of banks before we start evaluating robbery paths set arbitrarily.

A & D are fixed (set before algorithm begins and never changed), while B & C are dynamic (set at the beggining, but changed
by optimization)


HOW VARIABLES ARE USED:

B and C are the only values optimized at each iteration. Since:
- B represents vault worth, we will increase this factor at each iteration until we've found a number of banks bellow the 
  value assigned to D
- C represents time spent, we will decrease this factor at each iteration until we've found a number of banks bellow the
  value assigned to D

"""

"""
Global variables that can be tweaked:
"""
# A
# in h (hour)
time_threshold_for_edge_to_be_considered = 0.5
# B
# The factor by which we multiply the mean of the values of bank vaults
# in order to get the min threshold to consider a bank as a node.
# The higher this value, the richer our banks need to be for us
# to consider them. mean_money_factor_modifier is the step-up value
# we use at each iteration optimization. these are in km.
mean_money_factor=2.9
mean_money_factor_modifier=0.03
# C
# The factor by which we multiply the mean of the values of time
# it takes to rob a bank in order to get the min threshold to 
# consider a bank as a node. The smaller this value, the more selective
# we are with our options. mean_time_factor_modifier is the step-up value
# we use at each iteration optimization. these are in h (hour).
mean_time_factor=0.5
mean_time_factor_modifier=0.014
# Max number of starting points to assign to each process
max_starting_points_per_process=2
# D
def max_number_of_banks_to_consider():
    return int(float(ideal_number_of_processes()) * max_starting_points_per_process)
# in h
maximum_amount_of_time_for_any_given_path=24.0
# Im km/h
speed_at_which_we_travel_between_banks=30


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
    t = dist / vel
    return None if t > maxT else t


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

"""

METHODS TO DECIDE HOW MANY PROCESSES TO CREATE
AND HOW MANY DFS STARTING POINTS TO PASS TO EACH

"""

"""
Defaults to leaving 1 core thread free 
for laptops with more than 4 available 
threads
"""
def ideal_number_of_processes():
    count = multiprocessing.cpu_count()
    if count <= 2: return 2
    return count - 1

"""
Spreads out the starting points from the origin across the number of processes
we'll be creating
lst: the list of neighbors to the origin
into: (Optional) int
"""
def split_list_for_multiprocess(lst, into=ideal_number_of_processes()):
    lstC = lst.copy()
    result = [[] for _ in range(into)]
    counter = 0
    while len(lstC) > 0:
        result[counter].append(lstC.pop(0))
        counter = 0 if counter == into - 1 else counter + 1
    return result


def groups_to_split_origin_neighbors_into(size_batch):
    if size_batch < 50: 
        if size_batch <= ideal_number_of_processes() * 2: return 2
        return 1
    if 50 < size_batch < 150: return 2
    else: return 1

"""
MARK - GRAPH CLASS
"""
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

    def new_edge(self, A, B, t):
        if self.G.has_edge(A, B) is False and self.G.has_edge(B, A) is False:
            self.G.add_edge(A, B, weight=t)
        if str(A) == 'o' and str(B) != 'o':
            self.origin_neighors.add(B)
        if str(A) != 'o' and str(B) == 'o':
            self.origin_neighors.add(A)

    def longest_path(self):
        return nx.dag_longest_path(self.G)

    def neighbors_of_origin(self):
        return list(self.origin_neighors)

    """
    The method we pass to each worker process
    than: max weight of a path
    grouping: the group of origin's neighbors to work from
    update_counts: a convenience method to print how many origin-neighbor groups we've completed
    """
    def execute_sub_grouping(self, than, grouping, update_counts):
        results = []
        """
        The depth-first-search method
        A: point we're at
        B: counter-point on the edge we're evaluating
        totat: total time spent on path so far
        amount: total amount stolen so far
        nodes: the hyphen-delimited string indicating nodes already visited (a-b-c-d-e..)
        """
        def run_edge(A, B, total=0,  amount=0, nodes=''):
            edge = self.G.get_edge_data(A,B)
            bWeight = self.G.nodes[B].get('node_weight', 1)
            next_total = edge['weight'] + bWeight + total
            # We sum up the money accumulated plus value of B
            next_amount = amount + self.G.nodes[B]['money']
            """
            If this sum remains unchanged, we've hit a 
            dead-end and should not proceed further than
            A
            """
            if next_total == total:
                results.append((nodes, amount))
                return
            elif next_total < than: #otherwise continue
                """
                assign B's neighbors to a set, only iterate beyond B
                on its neighbors not already visited, "successors"
                """
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
            """
            We run the dfs method on the origin's neighbor
            then update the count of total neighors completed
            """
            run_edge('o', item, total=0.0, nodes="o")
            update_counts()
        return results


    def paths_with_time_less(self, than, neighs_arg=None):
        """
        This is our bottleneck
        """
        neighs = neighs_arg if neighs_arg is not None else self.neighbors_of_origin()
        results_sublists = []
        # print(len(neighs), "neighbors to origin")
        print("The exit helicopter zone can be reached from {neig} neighboring banks".format(neig=len(neighs)))
        divided_neighbors = split_list_for_multiprocess(
            neighs
        )
        # print("Spliting up the neighbors into groups of {spli}".format(spli=groups_to_split_origin_neighbors_into(len(neighs))))
        print("These are the divided neighbors to origin accross {cpus} processes:".format(cpus=ideal_number_of_processes()))
        print(divided_neighbors)
        print("And we have {edges} edges to evaluate".format(edges=len(self.G.edges)))
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
            speed_at_which_we_travel_between_banks
        )
        return -1 if edge is None else edge
    df['or'] = df.apply(lambda_origin, axis=1)

"""
Method to filter out banks that meet our current optimization filters

df: [Dataframe]
"""
def filter_out_banks_with_means(df):
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
Method that only applies edges that meet our time constraint over time spent traveling
between banks

We iterate over every 2-bank combinations, and evaluate wether the time spent over the
distance seperating them meets our optimization criteria.

consider_distance_as_edge returns None if an edge does not meet this criteria, this
edge is then added to our networkX graph if the value returned is not None

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
            time_threshold_for_edge_to_be_considered, 
            speed_at_which_we_travel_between_banks
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
    sum_money = paths[0][1]
    print("total money robbed", sum_money)

    # We rebuild the array of integers from the reversed array of strings
    # since we started our path analyses from the extraction point
    return [int(i) for i in first][::-1]

"""
input:
------
df: [string]: path to the file

output:
-------
result: [df: Dataframe, path: int[]]
"""
def robber_algorithm(path):

    # First step
    df = pd.read_csv(path)
    determine_edges_to_origin(df)
    graph = Graph()

    print("The maximum number of neigboring banks", 
        "to the origin that can consider in 3 minutes is:",
        max_number_of_banks_to_consider()
    )
    


    def optimize_factors():
        global mean_time_factor
        global mean_money_factor
        to_use = filter_out_banks_with_means(
            df
        )
        # Second step applied as many times as needed
        iterate_and_apply_valid_edges(to_use, df, graph)
        # Where we check to see if our pool of banks meets our boundaris to be able to
        # complete this analysis in under 3 minutes
        if len(graph.neighbors_of_origin()) < max_number_of_banks_to_consider(): return
        else:
            print("Optimizing our selection criteria to narrow our options..")
            print(" ")
            print(" ")
            print(" ")
            print(" ")
            mean_time_factor -= mean_time_factor_modifier
            mean_money_factor += mean_money_factor_modifier
            graph.new_graph()
            return optimize_factors()

    optimize_factors()
    # Third step
    paths = graph.paths_with_time_less(maximum_amount_of_time_for_any_given_path)
    # Fourth and Fifth
    result = return_best_path_of_banks(paths)
    print("The ideal path would be :")
    print(result)
    print("Checking result..")
    return result

path_to_file='bank_data.csv'
result=robber_algorithm(path_to_file)
check_solution(result, pd.read_csv(path_to_file), speed_at_which_we_travel_between_banks)

