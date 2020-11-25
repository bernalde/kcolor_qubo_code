import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import bmat, diags, eye, hstack, identity, triu, vstack
from six import iteritems
from itertools import combinations


def nonlinear(Y, weight='weight', k=2, c1=2.0, c2=2.0, draw=False):
    """Return the QUBO with ground states corresponding to a nonlinear max k subgraph coloring of the graph.
    For each node i in the graph and each r color of k possible, we assign a boolean variable x_ir, where x_ir = 1 when i
    is colored with r and x_ir = 0 otherwise.
    This problem has two constraints, that two adjacent nodes i and j in the graph, befined by having an edge ij between them,
    cannot have the same color r, and that a node can at most have one color. Each constraint is formulated as a bilinear product
    and dualized with coefficients c1 and c2, respectively.
    We call the matrix defining our QUBO problem Q.


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
        Parameter to penalize that no two edges share the same color linear constraint (has to be > 1 see Theorem 2 in paper).

    c2: float, optional (default 2.0)
        Parameter to penalize that a node cannot have two colors linear constraint (has to be > 1 see Theorem 2 in paper).

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

    # Start by removing self-edges (learnt the hard way)
    Y.remove_edges_from(nx.selfloop_edges(Y))

    # Constructing the QUBO
    cost = dict(Y.nodes(data=weight, default=1))
    # Objective function -sum_x^2 = -sum_x
    Q = {('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r)): -
         cost[i] for i in Y.nodes for r in range(1, k+1)}
    for r in range(1, k+1):

        # First constraint summed over all edges ij and colors r: x_ir x_jr
        for edge in Y.edges():
            (i, j) = tuple(sorted(edge))
            Q[('x_%s_%s' % (i, r), 'x_%s_%s' % (j, r))] = c1

    # Second constraint summed over all nodes i: (sum_{r =/= p; r, p \in [k]} x_ir x_ip
    for i in Y.nodes():
        for rs in combinations(range(1, k+1), 2):
            Q[('x_%s_%s' % (i, rs[0]), 'x_%s_%s' % (i, rs[1]))] = c2

    # Positive offset because of optimization direction (minimization)
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
    """Return the QUBO with ground states corresponding to a linear max k subgraph coloring of the graph.
    For each node i in the graph and each r color of k possible, we assign a boolean variable x_ir, where x_ir = 1 when i
    is colored with r and x_ir = 0 otherwise.
    This problem has two constraints, that two adjacent nodes i and j in the graph, befined by having an edge ij between them,
    cannot have the same color r, and that a node can at most have one color. Each constraint is originally an inequality but through 
    binary slack variables s_ijr and t_i are converted to equalities and dualized with coefficients c1 and c2, respectively.
    We call the matrix defining our QUBO problem Q.

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
    # Start by removing self-edges (learnt the hard way)
    Y.remove_edges_from(nx.selfloop_edges(Y))

    # Constructing the QUBO
    cost = dict(Y.nodes(data=weight, default=1))
    # Objective function -sum_x^2 = -sum_x
    Q = {('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r)): -
         cost[i] for i in Y.nodes for r in range(1, k+1)}
    for r in range(1, k+1):
        # First constraint summed over all edges ij and colors r: (x_ir + x_jr + s_ijr - 1)^2 = x_ir^2 + x_jr^2 + s_ijr^2 + 1 + 2 x_ir x_jr + 2 x_ir s_ijr + 2 x_jr s_ijr - 2x_ir - 2x_jr - 2s_ijr
        # = -x_ir - x_jr - s_ijr + 2 x_ir x_jr + 2 x_ir s_ijr + 2 x_jr s_ijr + 1
        for edge in Y.edges():
            (i, j) = tuple(sorted(edge))
            Q[('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r))] -= c1
            Q[('x_%s_%s' % (j, r), 'x_%s_%s' % (j, r))] -= c1
            Q[('s_%s_%s_%s' % (i, j, r), 's_%s_%s_%s' % (i, j, r))] = -c1
            Q[('x_%s_%s' % (i, r), 'x_%s_%s' % (j, r))] = 2 * c1
            Q[('x_%s_%s' % (i, r), 's_%s_%s_%s' % (i, j, r))] = 2 * c1
            Q[('x_%s_%s' % (j, r), 's_%s_%s_%s' % (i, j, r))] = 2 * c1

    # Positive offset because of optimization direction (minimization)
    offset = k * c1 * Y.number_of_edges()

    # Second constraint summed over all nodes i: (sum_x + t - 1)^2 = (sum_x)^2 + 2 sum_x t - 2 sum_x + t^2 - 2 t + 1 = sum_x^2 + 2*comb(2,x) + 2 sum_x t - 2 sum_x - t + 1
    # = [ - sum_x + 2*comb(2,x) ] + 2 sum_x t - t + 1
    # Important comment here (x_1 + ... x_n)^2 = \sum_{0<=i<=n} \sum_{0<=j<=n} x_ix_j = x_1^2 + ... + x_n ^2 + 2*(x_1x_2 + x_1_x3 + ... + x_2x_3 + ... x_{n-1}x_n)
    for i in Y.nodes():
        for r in range(1, k+1):
            # First term of the squared sum (1st term)
            Q[('x_%s_%s' % (i, r), 'x_%s_%s' % (i, r))] -= c2
            # 2nd term
            Q[('x_%s_%s' % (i, r), 't_%s' % (i))] = 2 * c2
        # Second term of the squared sum (1st term)
        for rs in combinations(range(1, k+1), 2):
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

    # Y = nx.cycle_graph(3)
    # Y, alpha = devil_graphs(2)

    # Example random_5_0.25_0 showing CPLEX imlpementation converges to suboptimal 4 when true optimal is 5 with k=2
    # edges = [(0, 1), (0, 0), (1, 4), (1, 1), (4, 4), (2, 2), (3, 3)]

    Y = nx.Graph()
    Y.add_edges_from(edges)

    print("Linear reformulation")
    Q_l, l_offset = linear(Y, k=2, c1=2.0, c2=2.0, draw=True)
    print(Q_l)
    print(l_offset)

    print("Nonlinear reformulation")
    Q_n, n_offset = nonlinear(Y, k=2, c1=2.0, c2=2.0, draw=True)
    print(Q_n)
    print(n_offset)

    enumerate_solution = False

    if enumerate_solution:
        import dimod

        exact_sampler = dimod.reference.samplers.ExactSolver()
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q_l, offset=l_offset)

        # Fixing solution from CPLEX yield suboptimal (4) solution
        # bqm.fix_variable('x_0_2', 1)
        # bqm.fix_variable('x_1_1', 0)
        # bqm.fix_variable('x_1_2', 0)
        # bqm.fix_variable('x_2_1', 1)
        # bqm.fix_variable('x_3_2', 1)
        # bqm.fix_variable('x_4_2', 1)

        response = exact_sampler.sample(bqm)
        energies = response.data_vectors['energy']
        print(energies)
        print(min(energies))
        for ans in response.data(fields=['sample', 'energy'], sorted_by='energy'):
            print(ans)
            break
