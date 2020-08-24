import networkx as nx
import numpy as np
from scipy.sparse import hstack, vstack, eye, diags, identity, bmat
import matplotlib.pyplot as plt


def laserre(Y, y=None, draw=False):
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
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases are negative.

    Args:

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        y: Parameter to dualize constraints in Lasserre's approach. If none provided y = 2n+1 with n = |V|

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
    e2 = np.ones((B.shape[0], 1))  # Ones of size |Edges|

    # Positive matrix since we are minimizing
    mat = y*bmat([[np.transpose(B) * B, np.transpose(B)],
                [B, eye(B.shape[0])]])

    # Negative vector since we are minimizing
    vec = -bmat([[np.transpose(e1) + 2 * y * (np.transpose(e2) * np.transpose(B)),
                  2 * y * np.transpose(e2).flatten()]])

    # Move terms from the diagonal of the matrix and add them to the vector
    biases = dict(enumerate(mat.diagonal()))
    for k in biases.keys():
        biases[k] = biases[k] +  vec.toarray()[0][k]

    # Remove elements from the matrix's diagonal
    mat.setdiag(0)
    mat.eliminate_zeros()

    # We duplicate the upper-triangular part of the matrix since the graph is undirected
    Z = nx.from_scipy_sparse_matrix(2 * triu(mat), edge_attribute='bias')

    # We set the value of the vector as biases on the nodes
    nx.set_node_attributes(Z, biases, 'bias')

    offset = - y*B.shape[0]

    if draw:
        plt.figure()
        nx.draw(Z)
        plt.show()

    return Z, offset


def proposed(Y, M=2, draw=False):
    """
    proposed(Y):
    Returns a graph defined by the proposed reformulation of the stable set problem of a given initial graph Y(V,E).
    The problem becomes: max e1'*x - x'*A*x, where x is the vector of binary variables being the indicators of the nodes
    x belonging to the stable set (size |V|).
    The reformulation follows: A being the adjacency matrix of graph Y, the linear term in this reformulation is given by
     the vector (where e's are the one vector, e1 of length |V|

    After computing the adjacency matrix and the linear vector and the bias term we generate a NetworkX graph
    The graph returning this reformulation will have the adjacency matrix values as biases in the edges and the linear
    terms as biases in the nodes. Since DWave minimizes the Hamiltonians, the biases are negative.

    Args:

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        y: Parameter to dualize constraints in Lasserre's approach. If none provided y = 2n+1 with n = |V|

        draw: Boolean variable to visualize the produced graph

    Returns:

        Z: NetworkX undirected graph defined by the adjacency matrix D with biases in edges and nodes

        offset: Extra term to add to the optimal objective function
    """

    A = nx.adjacency_matrix(Y)

    e1 = np.ones((A.shape[0], 1))  # Ones of size |Nodes|

    # Positive matrix since we are minimizing
    mat = M*A

    # Negative vector since we are minimizing
    vec = -e1

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

    if draw:
        plt.figure()
        nx.draw(Z)
        plt.show()

    return Z