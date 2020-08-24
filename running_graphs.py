import networkx as nx
import numpy as np
from scipy.sparse import hstack, vstack, eye, diags, identity, bmat, triu
import matplotlib.pyplot as plt
import dwave.embedding
from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
import dimod
import dwave_networkx as dnx
import minorminer as mm

# Generating the butterfly graph with 5 nodes
n = 5
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (3, 4, 1.0), (4, 2, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E, weight='bias')

plt.figure()
nx.draw(G, with_labels=True)
plt.show()

X = dnx.chimera_graph(2, 2)

embedding = mm.find_embedding(G, X)

plt.figure()
dnx.draw_chimera_embedding(X, embedding, show_labels=True)

bqm = dimod.BinaryQuadraticModel.from_networkx_graph(G, 'BINARY')
print('QUBO Quadratic part')
print(bqm.quadratic)
print('QUBO Linear part')
print(bqm.linear)

sampler = ExactSolver()

print('Exact solver output')
responseq = sampler.sample(bqm)
for datum in responseq.data(['sample', 'energy', 'num_occurrences']):
    print(datum.sample, -datum.energy, "Occurrences: ", datum.num_occurrences)


# print('Dwave solver output')
# from dwave.system.samplers import DWaveSampler
# sampler = EmbeddingComposite(DWaveSampler())

# responseq = sampler.sample(bqm, num_reads=1000)
# for datum in responseq.data(['sample', 'energy', 'num_occurrences']):
#     print(datum.sample, -datum.energy, "Occurrences: ", datum.num_occurrences)
