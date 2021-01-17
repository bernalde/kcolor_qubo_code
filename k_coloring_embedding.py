import matplotlib.pyplot as plt
import random
import os

from k_coloring_formulations import linear, nonlinear
from devil_graphs import devil_graphs

import dimod
from dwave.embedding import chimera, pegasus
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
import ast

DRY_RUN = False


def embedding(instance, TEST, k=2, prob=0.25, seed=42,
              K=1, sampler='Advantage_system1.1', no_embed=False):

    # Store the results in csv files
    dir_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(dir_path, "results/embedding/")

    if sampler == 'Advantage_system1.1':
        chip = 'pegasus'
    else:
        chip = 'chimera'

    if instance == 'spreadsheet':
        # spreadsheet_name = "erdos_" + \
        #     str(int(100*prob)) + '_graphs_with_opt.xlsx'
        # spreadsheet_name = "erdos_" + \
        #     str(int(100*prob)) + '_graphs.xlsx'
        spreadsheet_name = "erdos_" + \
            str(int(100*prob)) + '_graphs_' + str(K) + '.xlsx'
        spreadsheet_name = os.path.join(results_path, spreadsheet_name)
        input_data = pd.read_excel(spreadsheet_name)
        n0 = 0
        N = input_data.shape[0]
        file_name = instance + str(int(100*prob)) + \
            '_embedding_' + str(k) + '_' + chip + '_' + str(K)
        K = 1
    elif instance == 'erdos':
        # Parameters for random graphs
        N = 51
        n0 = 3
        K = 5  # number of graphs
        seed = 1
        instance = "erdos_" + str(int(100*prob))
        file_name = instance + '_embedding_' + chip
    elif instance == 'cycle':
        # Parameters for cycles
        N = 200
        n0 = 100
        file_name = instance + '_embedding_' + chip
    elif instance == 'devil':
        # Parameters for devil graphs
        N = 10
        n0 = 8
        file_name = instance + '_embedding_' + chip
    else:
        print("Graph type not implemented")
        return()

    # If results directory does not exist, we create it
    if not(os.path.exists(results_path)):
        print('Results directory ' + results_path +
              ' does not exist. We will create it.')
        os.makedirs(results_path)

    file_name = os.path.join(results_path, file_name)

    # time horizons and time limit in seconds
    if TEST:
        random.seed(seed)
        time_limit = 60
        # Number of times the heuristic is run
        n_heur = 100
    else:
        time_limit = 3600
        # Number of times the heuristic is run
        n_heur = 1000

    if no_embed:
        n_heur = 0

    columns = ['id', 'n_nodes', 'n_edges', 'nodes',
               'edges', 'alpha', 'target', 'density_target']
    results = pd.DataFrame(columns=columns)

    # Graph corresponding to D-Wave 2000Q or Advantage_system1.1
    qpu = DWaveSampler(solver=sampler)
    qpu_edges = qpu.edgelist
    qpu_nodes = qpu.nodelist
    # X = dnx.chimera_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    # X = dnx.pegasus_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    X = nx.Graph()
    X.add_nodes_from(qpu_nodes)
    X.add_edges_from(qpu_edges)
    # nx.write_edgelist(X, os.path.join(results_path,"X.edgelist"))

    for index in range(K):
        for n in range(n0, N):
            print(n)
            temp = dict()

            # Target graph statistics
            temp['target'] = qpu.solver.id
            temp['density_target'] = nx.density(X)

            # Graph generation
            if instance == "spreadsheet":
                # Generate graph from spreadsheet
                Input = nx.Graph()
                nodes = ast.literal_eval(input_data.nodes[n])
                edges = ast.literal_eval(input_data.edges[n])
                Input.add_nodes_from(nodes)
                Input.add_edges_from(edges)
                alpha = 0
            elif instance == "cycle":
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
                temp['id'] = "random_" + \
                    str(n) + "_" + str(prob) + "_" + str(index)
                alpha = 0

            # Input graph parameters
            temp['alpha'] = alpha
            temp['n_nodes'] = Input.number_of_nodes()
            temp['n_edges'] = Input.number_of_edges()
            temp['nodes'] = Input.nodes()
            temp['edges'] = Input.edges()

            if DRY_RUN:
                pass
            else:

                # Problem reformulations
                # Nonlinear and Linear (Laserre) reformulation
                reforms = ['n', 'l']
                for ref in reforms:
                    if ref == 'n':
                        Q, offset = nonlinear(Input, k=k, draw=False)
                    elif ref == 'l':
                        Q, offset = linear(Input, k=k, draw=False)

                    # Graphs generation
                    bqm = dimod.BinaryQuadraticModel.from_qubo(
                        Q, offset=offset)
                    edges = list(itertools.chain(
                        bqm.quadratic, ((v, v) for v in bqm.linear)))

                    G = nx.Graph()
                    G.add_edges_from(edges)

                    # Writing edgelists for exact methods, not required right now
                    # nx.write_edgelist(G, "results/embedding/G.edgelist")

                    # Graph statistics
                    temp['nodes_' + ref] = G.nodes()
                    temp['edges_' + ref] = G.edges()
                    temp['n_nodes_' + ref] = G.number_of_nodes()
                    temp['n_edges_' + ref] = G.number_of_edges()
                    temp['density_' + ref] = nx.density(G)

                    # Heuristic method solutions
                    h_times = []
                    # h_embeds = []
                    h_lengths = []

                    fail = 0
                    tot_count = 0
                    min_length = X.number_of_nodes()
                    best_time = time_limit
                    best_embed = {}
                    if not no_embed:
                        for n_h in range(n_heur):
                            start = time.time()
                            h_embed = mm.find_embedding(
                                G, X, timeout=time_limit, random_seed=n_h)
                            end = time.time()
                            h_time = end - start
                            h_times.append(h_time)
                            # h_embeds.append(h_embed)
                            count = 0
                            for _, value in h_embed.items():
                                count += len(value)
                            tot_count += count
                            if count == 0:
                                fail += 1
                            elif count < min_length:
                                best_embed = h_embed
                                min_length = count
                                best_time = h_time
                            h_lengths.append(count)
                        succ = (n_heur - fail) / n_heur
                        # temp['heur_embeds_' + ref] = h_embeds
                        temp['heur_times_' + ref] = h_times
                        temp['heur_lengths_' + ref] = h_lengths
                        temp['heur_succ_' + ref] = succ
                        if succ == 0:
                            temp['heur_avgl_' + ref] = 'NaN'
                            temp['heur_stdevl_' + ref] = 'NaN'
                        else:
                            avgl = tot_count / (succ * n_heur)
                            temp['heur_avgl_' + ref] = avgl
                            temp['heur_stdevl_' +
                                ref] = statistics.stdev(h_lengths)
                        temp['heur_avgt_' + ref] = np.median(h_times)
                        temp['heur_best_embed_' + ref] = best_embed
                        temp['heur_best_length_' + ref] = min_length
                        temp['heur_best_time_' + ref] = best_time

                    # start = time.time()
                    # Fully connected graph embedding happening here
                    # if len(Q) <= 63:
                    # if len(Q) <= 128:
                    # full_embed = chimera.find_clique_embedding(
                    #     len(Q), 16, target_edges=qpu_edges)
                    # full_embed = pegasus.find_clique_embedding(
                    #     len(Q), m=16)
                    # end = time.time() - start
                    # count = 0
                    # tot_count = 0
                    # for _, value in full_embed.items():
                    #     count += len(value)
                    # tot_count += count

                    # temp['full_embed_' + ref] = full_embed
                    # temp['full_length_' + ref] = tot_count
                    # temp['full_time_' + ref] = end
                    # else:
                    # temp['full_embed_' + ref] = 'NaN'
                    # temp['full_length_' + ref] = 'NaN'
                    # temp['full_time_' + ref] = 'NaN'

            results = results.append(temp, ignore_index=True)

            results.to_csv(file_name + ".csv")

    sol_total = pd.DataFrame.from_dict(results)

    sol_total.to_excel(os.path.join(results_path, file_name + ".xlsx"))


if __name__ == "__main__":

    # graph_type = 'erdos'
    # graph_type = 'cycle'
    # graph_type = 'devil'
    graph_type = 'spreadsheet'
    k = 2

    TEST = True
    no_embed = True
    prob = 0.75  # graph probability
    K = 4

    sampler = 'DW_2000Q_6'
    # sampler = 'Advantage_system1.1'

    embedding(instance=graph_type, TEST=TEST, k=k,
              prob=prob, K=K, sampler=sampler, no_embed=no_embed)
