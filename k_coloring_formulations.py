import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import bmat, diags, eye, hstack, identity, triu, vstack
from six import iteritems
from itertools import combinations


def nonlinear(Y, weight='weight', M=2.0, k=2, draw=False):
    """ Return the QUBO with ground states corresponding to a 


    Parameters
    ----------
    Y: NetworkX Graph

    weight : string, optional (default 'weight')
        If None, every node has equal weight. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have max weight.

    M: float, optional (default 2.0)
        Parameter to penalize constraints in proposed approach (has to be > 1).

    k: int, optional (default 2)
        Parameter to determine the coloring of the number. With k=1 reduces to stable set.

    draw: bool
        Boolean variable to visualize the produced graph

    Returns:

        Q: dict
            The QUBO with ground states corresponding to a maximum weighted independent set.

        offset: float
            Extra term to add to the optimal objective function
    """

    # empty QUBO for an empty graph
    if not Y:
        return {}

    # We assume that the sampler can handle an unstructured QUBO problem, so let's set one up.
    # Let us define the largest independent set to be S.
    # For each node n in the graph, we assign a boolean variable v_n, where v_n = 1 when n
    # is in S and v_n = 0 otherwise.
    # We call the matrix defining our QUBO problem Q.
    # On the diagonal, we assign the linear bias for each node to be the negative of its weight.
    # This means that each node is biased towards being in S.
    # Negative weights are considered 0.
    # On the off diagonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.

    # Start by removing self-edges (learnt the hard way)
    Y.remove_edges_from(nx.selfloop_edges(Y))

    cost = dict(Y.nodes(data=weight, default=1))
    Q = {(node, node): -cost[node] for node in Y}
    Q.update({edge: M for edge in Y.edges})

    offset = 0.0

    if draw:
        Z = nx.Graph()
        Z.add_nodes_from(((u, {'weight': bias})
                          for (u, u), bias in iteritems(Q)))
        Z.add_edges_from(((u, v, {'weight': bias})
                          for (u, v), bias in iteritems(Q)))
        pos = nx.spring_layout(Z)
        edge_labels = dict([((u, v), d['weight'])
                            for u, v, d in Z.edges(data=True)])
        plt.figure()
        nx.draw(Z, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(Z, pos=pos, edge_labels=edge_labels)
        plt.show()

    return Q, offset


def linear(Y, weight='weight', k=2, c1=2.0, c2=2.0, draw=False):
    """Return the QUBO with ground states corresponding to a

    Parameters
    ----------
    Y: NetworkX Graph

    weight : string, optional (default 'weight')
        If None, every node has equal weight. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have max weight.

    k: int, optional (default 2)
        Parameter to determine the coloring of the number. With k=1 reduces to stable set.

    c1: float, optional (default 2.0)
        Parameter to penalize that no two edges share the same color linear constraint (has to be > 1 see Theorem 1 in paper).

    c2: float, optional (default 2.0)
        Parameter to penalize that a node cannot have two colors linear constraint (has to be > 1 see Theorem 1 in paper).

    draw: bool
        Boolean variable to visualize the produced graph

    Returns
    --------
    Q: dict, with tuple keys and float values
        Dictionary that specifies the Q matrix in the Binary Quadratic Model min x'Qx + c for DWave

    offset: float
        Extra term c to add to in the Binary Quadratic Model min x'Qx + c
    """

    # empty QUBO for an empty graph
    if not Y:
        return {}

    # We assume that the sampler can handle an unstructured QUBO problem, so let's set one up.
    # Let us define the largest independent set to be S.
    # For each node n in the graph, we assign a boolean variable v_n, where v_n = 1 when n
    # is in S and v_n = 0 otherwise.
    # We call the matrix defining our QUBO problem Q.
    # On the diagonal, we assign the linear bias for each node to be the negative of its weight.
    # This means that each node is biased towards being in S.
    # Negative weights are considered 0.
    # On the off diagonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.

    # Start by removing self-edges (learnt the hard way)
    Y.remove_edges_from(nx.selfloop_edges(Y))

    # Constructing the QUBO
    cost = dict(Y.nodes(data=weight, default=1))
    # Objective function -sum_x^2 = -sum_x
    Q = {('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r)): -
         cost[i] for i in Y.nodes for r in range(1, k+1)}
    for r in range(1,k+1):
        # First constraint summed over all edges and colors (xi + xj + sij - 1)^2 = xi^2 + xj^2 + sij^2 + 1 + 2 xi xj + 2 xi sij + 2 xj sij - 2x1 - 2xj - 2sij 
        # = -xi - xj - sij + 2 xi xj + 2 xi sij + 2 xj sij + 1
        for edge in Y.edges():
            (u, v) = tuple(sorted(edge))
            Q[('x_%s_%s' % (u, r), 'x_%s_%s' % (u, r))] -= c1
            Q[('x_%s_%s' % (v, r), 'x_%s_%s' % (v, r))] -= c1
            Q[('s_%s_%s_%s' % (u, v, r), 's_%s_%s_%s' % (u, v, r))] = -c1
            Q[('x_%s_%s' % (u, r), 'x_%s_%s' % (v, r))] = 2 * c1
            Q[('x_%s_%s' % (u, r), 's_%s_%s_%s' % (u, v, r))] = 2 * c1
            Q[('x_%s_%s' % (v, r), 's_%s_%s_%s' % (u, v, r))] = 2 * c1

    # Positive offset because of optimization direction (minimization)
    offset = k * c1 * Y.number_of_edges()

    # Second constraint summed over all i (sum_x + t - 1)^2 = (sum_x)^2 + 2 sum_x t - 2 sum_x + t^2 - 2 t + 1 = sum_x^2 + 2*comb(2,x) + 2 sum_x t - 2 sum_x - t + 1
    # = [ - sum_x + 2*comb(2,x) ] + 2 sum_x t - t + 1
    # Important comment here (x_1 + ... x_n)^2 = \sum_{0<=i<=n} \sum_{0<=j<=n} x_ix_j = x_1^2 + ... + x_n ^2 + 2*(x_1x_2 + x_1_x3 + ... + x_2x_3 + ... x_{n-1}x_n)
    for i in Y.nodes():
        for r in range(1,k+1):
            # First term of the squared sum (1st term)
            Q[('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r))] -= c2
            # 2nd term
            Q[('x_%s_%s' % (i, r), 't_%s' % (i))] = 2 * c2
        # Second term of the squared sum (1st term)
        for rs in combinations(range(1,k+1),2):
            Q[('x_%s_%s' % (i, rs[0]), 'x_%s_%s' % (i, rs[1]))] = 2 * c2
        # 3rd term
        Q[('t_%s' % (i), 't_%s' % (i))] = -c2

    # Positive offset because of optimization direction (minimization)
    offset += c2 * Y.number_of_nodes()     

    if draw:
        Z = nx.Graph()
        Z.add_nodes_from(((u, {'weight': bias})
                          for (u, u), bias in iteritems(Q)))
        Z.add_edges_from(((u, v, {'weight': bias})
                          for (u, v), bias in iteritems(Q)))
        pos = nx.spring_layout(Z)
        edge_labels = dict([((u, v), d['weight'])
                            for u, v, d in Z.edges(data=True)])
        plt.figure()
        nx.draw(Z, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(Z, pos=pos, edge_labels=edge_labels)
        plt.show()

    return Q, offset


if __name__ == "__main__":
    from devil_graphs import devil_graphs
    # import dimod

    # Y = nx.cycle_graph(3)
    # Y, alpha = devil_graphs(2)

    # Example random_5_0.25_0 showing CPLEX imlpementation converges to suboptimal
    edges = [(0, 1), (0, 0), (1, 4), (1, 1), (4, 4), (2, 2), (3, 3)]
    Y = nx.Graph()
    Y.add_edges_from(edges)
    
    print("Linear reformulation")
    Q_l, l_offset = linear(Y, k=2, c1=2.0, c2=2.0, draw=True)
    print(Q_l)
    print(l_offset)

    exact_sampler = dimod.reference.samplers.ExactSolver()
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_l, offset=l_offset)

    # Fixing solution from CPLEX yield suboptimal (4) solution
    # bqm.fix_variable('x_0_2', 1)
    # bqm.fix_variable('x_1_1', 0)
    # bqm.fix_variable('x_1_2', 0)
    # bqm.fix_variable('x_2_1', 1)
    # bqm.fix_variable('x_3_2', 1)
    # bqm.fix_variable('x_4_2', 1)

    # response = exact_sampler.sample(bqm)
    # energies = response.data_vectors['energy']
    # print(energies)
    # print(min(energies))
    # for ans in response.data(fields=['sample', 'energy'], sorted_by='energy'):
    #     print(ans)
    #     break

    # print("Nonlinear reformulation")
    # Q_n, n_offset = nonlinear(Y, M=2.0, k=3, draw=True)
    # print(Q_n)
    # print(n_offset)
