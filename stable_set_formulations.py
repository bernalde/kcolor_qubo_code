import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import hstack, vstack, eye, diags, identity, bmat, triu
from six import iteritems


def proposed(Y, weight=None, M=2.0, draw=False):
    """ Return the QUBO with ground states corresponding to a maximum weighted independent set.
    The QUBO is defined by the proposed reformulation of the stable set problem of a given initial graph Y(V,E).
    The problem becomes: max e1'*x - x'*A*x, where x is the vector of binary variables being the indicators of the nodes
    x belonging to the stable set (size |V|).
    The reformulation follows: A being the adjacency matrix of graph Y, the linear term in this reformulation is given by
     the vector (where e's are the one vector, e1 of length |V|

    After computing the adjacency matrix and the linear vector and the bias term we generate a NetworkX graph
    The graph returning this reformulation will have the adjacency matrix values as biases in the edges and the linear
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases are negative.


    Parameters
    ----------
    Y: NetworkX Graph

    weight : string, optional (default None)
        If None, every node has equal weight. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have max weight.

    M: float, optional (default 2.0)
        Parameter to penalize constraints in proposed approach (has to be > 1).

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
    # This means that each node is biased towards being in S. Weights are scaled to a maximum of 1.
    # Negative weights are considered 0.
    # On the off diagonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.
    cost = dict(Y.nodes(data=weight, default=1))
    scale = max(cost.values())
    Q = {(node, node): min(-cost[node] / scale, 0.0) for node in Y}
    Q.update({edge: M for edge in Y.edges})

    offset = 0.0

    if draw:
        Z = nx.Graph()
        Z.add_nodes_from(((u, {'weight': bias}) for (u, u), bias in iteritems(Q)))
        Z.add_edges_from(((u, v, {'weight': bias}) for (u, v), bias in iteritems(Q)))
        pos = nx.spring_layout(Z)
        edge_labels = dict([((u, v), d['weight'])
                            for u, v, d in Z.edges(data=True)])
        plt.figure()
        nx.draw(Z, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(Z, pos=pos, edge_labels=edge_labels)
        plt.show()

    return Q, offset


def laserre(Y, weight=None, M=None, draw=False):
    """
    laserre_no_diag(Y):
    Returns a graph defined by the Lasserre reformulation of the stable set problem of a given initial graph Y(V,E).
    The problem becomes: max v'*X - X'*D*X + c, where X is the variables vector X = [x';s'] of binary variables x being
    the indicators of the nodes x belonging to the stable set (size |V|) and s being the slack variables for the
    constraints (size |E|).
    The reformulation follows: A being the adjacency matrix of graph Y, B being the incidence matrix of graph Y
    The adjacency matrix of the reformulation graph Z is:
    D = y*[[B'B B'],
           [B   I]];
    The linear term in this reformulation is given by the vector (where e's are the one vector, e1 of length |V|, e2 of
    length |E|)
    v = [e1+2*y*e2'*B
        2*y*e1]
    The offset term is in this case is c = -y*|E|

    After computing the adjacency matrix and the linear vector and the bias term we generate a NetworkX graph
    The graph returning this reformulation will have the adjacency matrix values as biases in the edges and the linear
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases and offset are negative.

    Args:

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        y: Parameter to dualize constraints in Lasserre's approach. If none provided y = 2n+1 with n = |V|

        draw: Boolean variable to visualize the produced graph

    Returns:

        Z: NetworkX undirected graph defined by the adjacency matrix D with biases in edges and nodes

        offset: Extra term to add to the optimal objective function
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
    # This means that each node is biased towards being in S. Weights are scaled to a maximum of 1.
    # Negative weights are considered 0.
    # On the off diagonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.

    if M is None:
        M = 2 * Y.number_of_nodes() + 1

    cost = dict(Y.nodes(data=weight, default=1))
    scale = max(cost.values())
    Q = {(node, node): min(-cost[node] / scale, 0.0) for node in Y}
    for (u, v) in Y.edges():
        Q[(u, v)] = 2 * M
        Q[(u, 's_%s_%s' % (u, v))] = 2 * M
        Q[(v, 's_%s_%s' % (u, v))] = 2 * M
        Q[('s_%s_%s' % (u, v), 's_%s_%s' % (u, v))] = -M
        Q[(u, u)] -= M
        Q[(v, v)] -= M

    # Positive offset because of optimization direction (minimization)
    offset = M * Y.number_of_edges()

    if draw:
        Z = nx.Graph()
        Z.add_nodes_from(((u, {'weight': bias}) for (u, u), bias in iteritems(Q)))
        Z.add_edges_from(((u, v, {'weight': bias}) for (u, v), bias in iteritems(Q)))
        pos = nx.spring_layout(Z)
        edge_labels = dict([((u, v), d['weight'])
                            for u, v, d in Z.edges(data=True)])
        plt.figure()
        nx.draw(Z, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(Z, pos=pos, edge_labels=edge_labels)
        plt.show()

    return Q, offset


def laserre_old(Y, y=None, draw=False):
    """
    laserre_no_diag(Y):
    Returns a graph defined by the Lasserre reformulation of the stable set problem of a given initial graph Y(V,E).
    The problem becomes: max v'*X - X'*D*X + c, where X is the variables vector X = [x';s'] of binary variables x being
    the indicators of the nodes x belonging to the stable set (size |V|) and s being the slack variables for the
    constraints (size |E|).
    The reformulation follows: A being the adjacency matrix of graph Y, B being the incidence matrix of graph Y
    The adjacency matrix of the reformulation graph Z is:
    D = y*[[B'B B'],
           [B   I]];
    The linear term in this reformulation is given by the vector (where e's are the one vector, e1 of length |V|, e2 of
    length |E|)
    v = [e1+2*y*e2'*B
        2*y*e1]
    The offset term is in this case is c = -y*|E|

    After computing the adjacency matrix and the linear vector and the bias term we generate a NetworkX graph
    The graph returning this reformulation will have the adjacency matrix values as biases in the edges and the linear
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases and offset are negative.

    Args:

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        M: Parameter to dualize constraints in Lasserre's approach. If none provided y = 2n+1 with n = |V|

        draw: Boolean variable to visualize the produced graph

    Returns:

        Z: NetworkX undirected graph defined by the adjacency matrix D with biases in edges and nodes

        offset: Extra term to add to the optimal objective function
    """

    A = nx.adjacency_matrix(Y)
    B = nx.incidence_matrix(Y)

    if y is None:
        y = 2 * (A.shape[0]) + 1

    e1 = np.ones((A.shape[0], 1))  # Ones of size |Nodes|
    e2 = np.ones((B.shape[1], 1))  # Ones of size |Edges|

    # Positive matrix since we are minimizing
    mat = y * bmat([[np.transpose(B) * B, np.transpose(B)],
                    [B, eye(B.shape[0])]])

    # Negative vector since we are minimizing
    vec = -bmat([[np.transpose(e1) + 2 * y * (np.transpose(e2) * np.transpose(B)),
                  2 * y * np.transpose(e2).flatten()]])

    # Move terms from the diagonal of the matrix and add them to the vector
    biases = dict(enumerate(mat.diagonal()))
    for k in biases.keys():
        biases[k] = biases[k] + vec.toarray()[0][k]

    # Remove elements from the matrix's diagonal
    mat.setdiag(0)
    mat.eliminate_zeros()

    # We duplicate the upper-triangular part of the matrix since the graph is undirected
    Z = nx.from_scipy_sparse_matrix(2 * triu(mat), edge_attribute='bias')

    # We set the value of the vector as biases on the nodes
    nx.set_node_attributes(Z, biases, 'bias')

    # Positive offset because of optimization direction (minimization)
    offset = y * B.shape[0]

    if draw:
        plt.figure()
        nx.draw(Z)
        plt.show()

    return Z, offset


def proposed_old(Y, M=2, draw=False):
    """ Return the QUBO with ground states corresponding to a maximum weighted independent set.
    The QUBO is defined by the proposed reformulation of the stable set problem of a given initial graph Y(V,E).
    The problem becomes: max e1'*x - x'*A*x, where x is the vector of binary variables being the indicators of the nodes
    x belonging to the stable set (size |V|).
    The reformulation follows: A being the adjacency matrix of graph Y, the linear term in this reformulation is given by
     the vector (where e's are the one vector, e1 of length |V|

    After computing the adjacency matrix and the linear vector and the bias term we generate a NetworkX graph
    The graph returning this reformulation will have the adjacency matrix values as biases in the edges and the linear
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases are negative.

    Args:

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        M: Parameter to peanlize constraints in proposed approach (has to be > 1). If none provided M=2

        draw: Boolean variable to visualize the produced graph

    Returns:

        Z: NetworkX undirected graph.
            Graph defined by the adjacency matrix D with biases in edges and nodes

        offset: float
            Extra term to add to the optimal objective function
    """

    A = nx.adjacency_matrix(Y)

    e1 = np.ones((A.shape[0], 1))  # Ones of size |Nodes|

    # Positive matrix since we are minimizing
    mat = M * A

    # Negative vector since we are minimizing
    vec = -e1

    # Move terms from the diagonal of the matrix and add them to the vector
    biases = dict(enumerate(mat.diagonal()))
    for k in biases.keys():
        biases[k] = biases[k] + vec.item(k)

    # Remove elements from the matrix's diagonal
    mat.setdiag(0)
    mat.eliminate_zeros()

    # We duplicate the upper-triangular part of the matrix since the graph is undirected
    Z = nx.from_scipy_sparse_matrix(2 * triu(mat), edge_attribute='bias')

    # We set the value of the vector as biases on the nodes
    nx.set_node_attributes(Z, biases, 'bias')

    offset = 0

    if draw:
        plt.figure()
        nx.draw(Z)
        plt.show()

    return Z, offset


if __name__ == "__main__":
    from devil_graphs import devil_graphs

    # Y = nx.cycle_graph(3)
    Y, alpha = devil_graphs(2)
    # nx.draw(Y)
    # plt.show()

    # print("Lasserre's reformulation")
    # L, l_offset = laserre_old(Y, draw=False)
    # print(L.edges())
    # print(L.nodes())
    # print(nx.adjacency_matrix(L).todense())
    # print(l_offset)
    #
    # print("Proposed reformulation")
    # P, p_offset = proposed_old(Y, M=1.0, draw=False)
    # print(P.edges())
    # print(P.nodes())
    # print(nx.adjacency_matrix(P).todense())
    # print(p_offset)

    print("Lasserre reformulation")
    Q, l_offset = laserre(Y, draw=True)
    print(Q)
    print(l_offset)

    print("Proposed reformulation")
    Q, p_offset = proposed(Y, M=1.0, draw=True)
    print(Q)
    print(p_offset)
