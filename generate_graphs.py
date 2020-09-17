import matplotlib.pyplot as plt
import random
import os

from stable_set_formulations import laserre, proposed
from devil_graphs import devil_graphs

import dimod
from dwave.embedding import chimera
from dwave.system import DWaveSampler
import dwave_networkx as dnx

import subprocess
import time
import statistics
# import find_embedding
import minorminer as mm
import networkx as nx
import numpy as np
import pandas as pd
import itertools

TEST = True

# Parameters for random graphs
N = 51
n0 = 3
prob = 0.75  # graph probability
K = 5  # number of graphs
seed = 1
instance = "erdos_" + str(int(100*prob))

# Parameters for cycles
# N = 200
# n0 = 100
# instance = "cycle"

# Parameters for devil graphs
# N = 10
# n0 = 8
# instance = "devil"


def generate_graphs():

    # Store the results in csv files
    dir_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(dir_path, "results/embedding/")
    if not(os.path.exists(results_path)):
        print('Results directory ' + results_path +
              ' does not exist. We will create it.')
        os.makedirs(results_path)

    file_name = instance + '_graphs'
    file_name = os.path.join(results_path, file_name)

    # time horizons and time limit in seconds
    if TEST:
        random.seed(seed)
        time_limit = 120
        # Number of times the heuristic is run
        n_heur = 100
    else:
        time_limit = 3600
        # Number of times the heuristic is run
        n_heur = 1000

    columns = ['id', 'n_nodes', 'n_edges', 'nodes', 'edges', 'alpha', 'target', 'density_target']
    results = pd.DataFrame(columns=columns)

    # Graph corresponding to D-Wave 2000Q
    qpu = DWaveSampler()
    qpu_edges = qpu.edgelist
    qpu_nodes = qpu.nodelist
    X = dnx.chimera_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    nx.write_edgelist(X, os.path.join(results_path,"X.edgelist"))
    


    for k in range(K):
        for n in range(n0, N):
            print(n)
            temp = dict()

            # Target graph statistics
            temp['target'] = qpu.solver.id
            temp['density_target'] = nx.density(X)

            # Graph generation
            if instance == "cycle":
                # Cycle graphs
                Input = nx.cycle_graph(n)
                temp['id'] = "cycle_" + str(n)
                alpha = np.floor(n / 2)
            elif instance == "devil":
                # Devil graphs
                Input, alpha = devil_graphs(n)
                temp['id'] = "devil_rank_" + str(n)
            else:
                # Random graphs
                Input = nx.erdos_renyi_graph(n, prob)
                temp['id'] = "random_" + str(n) + "_" + str(prob) + "_" + str(k)
                alpha = 0

            # Input graph parameters
            temp['alpha'] = alpha
            temp['n_nodes'] = Input.number_of_nodes()
            temp['n_edges'] = Input.number_of_edges()
            temp['nodes'] = Input.nodes()
            temp['edges'] = Input.edges()


            # Problem reformulations
            # Proposed and Laserre reformulation
            reforms = ['p', 'l']
            for ref in reforms:
                if ref == 'p':
                    Q, offset = proposed(Input, M=1, draw=False)
                elif ref == 'l':
                    Q, offset = laserre(Input, draw=False)
                
                # Graphs generation
                bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
                edges = list(itertools.chain(bqm.quadratic, ((v, v) for v in bqm.linear)))

                G = nx.Graph()
                G.add_edges_from(edges)

                # Graph statistics
                temp['nodes_' + ref] = G.nodes()
                temp['edges_' + ref] = G.edges()
                temp['n_nodes_' + ref] = G.number_of_nodes()
                temp['n_edges_' + ref] = G.number_of_edges()
                temp['density_' + ref] = nx.density(G)


            results = results.append(temp, ignore_index=True)

            results.to_csv(file_name + ".csv")

    sol_total = pd.DataFrame.from_dict(results)

    sol_total.to_excel(os.path.join(results_path,file_name + ".xlsx"))


if __name__ == "__main__":
    generate_graphs()
