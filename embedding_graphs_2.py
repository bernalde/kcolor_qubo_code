import matplotlib.pyplot as plt
import random
from stable_set_formulations import laserre, proposed
from devil_graphs import devil_graphs
import dwave.embedding
import dimod
import random
import subprocess
import time
from collections import defaultdict

import dwave_networkx as dnx
#import find_embedding
import minorminer as mm
import networkx as nx
import numpy as np
import pandas as pd

random.seed(1)
solutions = defaultdict(list)
keys = ['id', 'n_nodes', 'n_edges', 'alpha', 'target', 'density_target', 'nodes_l', 'edges_l', 'n_nodes_l', 'n_edges_l', 'density_l', 'heur_times_l',
        'heur_embeds_l', 'heur_lengths_l', 'heur_succ_l', 'heur_avgl_l', 'heur_avgt_l', 'nodes_p', 'edges_p', 'n_nodes_p', 'n_edges_p', 'density_p', 'heur_times_p',
        'heur_embeds_p', 'heur_pengths_p', 'heur_succ_p', 'heur_avgl_p', 'heur_avgt_p']
solutions.fromkeys(keys, [])

X = dnx.chimera_graph(16, 16)
nx.write_edgelist(X, "results/X.edgelist")
#embedder = find_embedding.EmbeddingProblem()

# Parameters for cycles
# N = 65
# n0 = 4

# Parameters for devil graphs
N = 11
n0 = 10

for n in range(n0, N):
    print(n)

    # Graph generation
    # Cycle graphs
    # Input = nx.cycle_graph(n)
    # solutions['id'].append("cycle_" + str(n))

    # Devil graphs
    Input, alpha = devil_graphs(n)
    solutions['id'].append("devil_rank_" + str(n))
    solutions['alpha'].append(alpha)

    # Input graph parameters
    solutions['n_nodes'].append(Input.number_of_nodes())
    solutions['n_edges'].append(Input.number_of_edges())

    # Graphs reformulations
    L, l = laserre(Input, draw=False)
    P, p = proposed(Input, M=1, draw=False)

    nx.write_edgelist(L, "results/L.edgelist")
    nx.write_edgelist(P, "results/P.edgelist")

    # Lasserre graph statistics
    solutions['nodes_l'].append(L.nodes())
    solutions['edges_l'].append(L.edges())
    solutions['n_nodes_l'].append(L.number_of_nodes())
    solutions['n_edges_l'].append(L.number_of_edges())
    solutions['density_l'].append(nx.density(L))

    # Proposed graph statistics
    solutions['nodes_p'].append(P.nodes())
    solutions['edges_p'].append(P.edges())
    solutions['n_nodes_p'].append(P.number_of_nodes())
    solutions['n_edges_p'].append(P.number_of_edges())
    solutions['density_p'].append(nx.density(P))

    # Target graph statistics
    solutions['target'].append('chimera_44')
    solutions['density_target'].append(nx.density(X))

    n_heur = 100

    # Heuristic method solutions for Lasserre
    h_times = []
    h_embeds = []
    h_lengths = []

    fail = 0
    tot_count = 0
    for i in range(n_heur):
        start = time.time()
        h_embed = mm.find_embedding(L, X)
        end = time.time()
        h_time = end - start
        h_times.append(h_time)
        h_embeds.append(h_embed)
        count = 0
        for key, value in h_embed.items():
            count += len(value)
        tot_count += count
        if count == 0:
            fail += 1
        h_lengths.append(count)
    succ = (n_heur - fail) / n_heur
    solutions['heur_embeds_l'].append(h_embeds)
    solutions['heur_times_l'].append(h_times)
    solutions['heur_lengths_l'].append(h_lengths)
    solutions['heur_succ_l'].append(succ)
    if succ == 0:
        solutions['heur_avgl_l'].append('NaN')
    else:
        avgl = tot_count / (succ * n_heur)
        solutions['heur_avgl_l'].append(avgl)

    solutions['heur_avgt_l'].append(np.median(h_times))

    # Heuristic method solutions for Proposed
    h_times = []
    h_embeds = []
    h_lengths = []

    fail = 0
    tot_count = 0
    for i in range(n_heur):
        start = time.time()
        h_embed = mm.find_embedding(P, X)
        end = time.time()
        h_time = end - start
        h_times.append(h_time)
        h_embeds.append(h_embed)
        count = 0
        for key, value in h_embed.items():
            count += len(value)
        tot_count += count
        if count == 0:
            fail += 1
        h_lengths.append(count)
    succ = (n_heur - fail) / n_heur
    solutions['heur_embeds_p'].append(h_embeds)
    solutions['heur_times_p'].append(h_times)
    solutions['heur_lengths_p'].append(h_lengths)
    solutions['heur_succ_p'].append(succ)
    if succ == 0:
        solutions['heur_avgl_p'].append('NaN')
    else:
        avgl = tot_count / (succ * n_heur)
        solutions['heur_avgl_p'].append(avgl)

    solutions['heur_avgt_p'].append(np.median(h_times))


sol_total = pd.DataFrame.from_dict(solutions)

sol_total.to_excel("results/devil_big.xlsx")



'''
bqm = dimod.BinaryQuadraticModel.from_networkx_graph(L, 'BINARY')
bqm.add_offset(l)

from dimod.reference.samplers import ExactSolver
sampler = ExactSolver()
responseq = sampler.sample(bqm)
for datum in responseq.data(['sample', 'energy', 'num_occurrences']):
    print(datum.sample, -datum.energy, "Occurrences: ", datum.num_occurrences)


# from dwave.system.composites import EmbeddingComposite
# from dwave.system.samplers import DWaveSampler
# sampler = EmbeddingComposite(DWaveSampler())
# responseq = sampler.sample(bqm, num_reads=1000)
# for datum in responseq.data(['sample', 'energy', 'num_occurrences']):
#     print(datum.sample, -datum.energy, "Occurrences: ", datum.num_occurrences)
'''