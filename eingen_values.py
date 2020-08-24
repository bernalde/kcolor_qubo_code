import itertools
import math
import os
import sys
from time import time  # timing package

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
from devil_graphs import devil_graphs
from scipy import sparse
from scipy.sparse.linalg import eigs
from stable_set_formulations import laserre, proposed

four_node_edges = [[],
                   [(0, 1)],
                   [(0, 1), (0, 3)], [(0, 1), (2, 3)],
                   [(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (0, 3)], [(0, 1), (1, 2), (2, 0)],
                   [(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 1), (1, 2), (2, 0), (3, 0)],
                   [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
                   [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
                   ]

# for k in range(len(four_node_edges)):
for k in range(2, 3):
    G = nx.Graph()
    # G = nx.cycle_graph(k)
    # G.add_nodes_from([0, 1, 2, 3])
    # G.add_edges_from(four_node_edges[k])
    # G, alpha = devil_graphs(k)
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)])
    nx.draw(G, with_labels=True)
    plt.show()
    ti = time()  # start timer

    # Find the maximum independent set, which is known in this case to be of length 3
    Q, offset = proposed(G)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
    indices = dict((key, idx) for (idx, key) in enumerate(bqm.linear.keys()))

    # Number of qubits
    n = len(list(bqm.linear))
    print(n)

    # Pauli matrices

    Id = [[1, 0], [0, 1]]
    s_x = [[0, 1], [1, 0]]
    s_y = [[0, -1j], [1j, 0]]
    s_z = [[1, 0], [0, -1]]


    # A and B functions such that H = A(s)Hd + B(s)Hp. These are specific to D-Wave 2000Q at NASA.
    # These are old fits by Zoe we will instead use the data provided by DWave directly

    def a_function(s):
        return (2 / math.pi) * (
                math.exp(3.173265 * s) * (207.253 * (1 - s) ** 9 + 203.843 * (1 - s) ** 7 - 380.659 * (1 - s) ** 8))


    def b_function(s):
        return (2 / math.pi) * (0.341734 + 6.713285 * s + 32.9702 * s * s)


    # s_list = np.arange(0, 1.0001, 0.001)

    # List of s values you want to use, between 0 and 1

    df = pd.read_excel(r'09-1192A-C_DW_2000Q_2_1_processor-annealing-schedule.xlsx',
                       sheet_name='DW_2000Q_2_processor-annealing-')


    # Nested Kronecker that we need for the individual Pauli matrices
    # Performs a Kronecker product on a series of matrices all stored in the variable a

    def nested_kronecker(a):
        if len(a) == 2:
            return np.kron(a[0], a[1])
        else:
            return np.kron(a[0], nested_kronecker(a[1:]))


    # Builds a Pauli matrix acting on qubit j (variable mat is the Pauli matrix we want applied on j, s_x, s_y or s_z)

    def individual_pauli(mat, j, num_of_spins):
        ops = []
        for k in range(j):
            ops.append(Id)
        ops.append(mat)
        for k in range(num_of_spins - j - 1):
            ops.append(Id)
        return nested_kronecker(ops)


    # Build driver and problem Hamiltonians
    Hd = 0
    for i in range(n):
        Hd = np.add(Hd, individual_pauli(s_x, i, n))

    Hp = 0
    for pair in bqm.quadratic:
        Hp = np.add(Hp,
                    bqm.quadratic[pair] * individual_pauli(s_z, indices[pair[0]], n) * individual_pauli(s_z, indices[
                        pair[1]], n))

    for qubit in bqm.linear:
        Hp = np.add(Hp, bqm.linear[qubit] * individual_pauli(s_z, indices[qubit], n))
    # If your Ising Hamiltonian also has an external field you can add it with terms of the form individual_pauli(s_z, i, n)

    s_list = df['s']

    # df = df.loc[[i for j, i in enumerate(df.index) if j % 10 == 0]]

    tolerance = 1
    idx_start = 0
    idx_end = len(s_list)
    print(idx_end)
    idx_interval = 100
    decrease = 10

    eigenvals = []
    s_plot = []
    mingap_ss = []
    index_ss = []
    # Calculate full Hamiltonian for the s values in your list and get the eigenvalues and gap
    while idx_interval >= 1:
        for idx in range(idx_start, idx_end, idx_interval):
            if df.loc[idx, 's'] in mingap_ss or idx in index_ss:
                pass
            else:
                H = df.loc[idx, 'A(s) (GHz)'] * Hd + df.loc[idx, 'B(s) (GHz)'] * Hp
                if n >= 14:
                    sH = sparse.csc_matrix(H)
                    eig = eigs(sH, 10, which='SR', tol=1e-3, return_eigenvectors=False)
                else:
                    eig = la.eigvalsh(H)
                eig = np.sort(eig.real)
                eigenvals.append(eig)
                # np.append(eig, eigenvals, axis=0)
                s_plot.append(df.loc[idx, 's'])

        s_plot, eigenvals = (list(t) for t in zip(*sorted(zip(s_plot, eigenvals))))

        eigenvalues = np.array(eigenvals)

        for i in range(1, eigenvalues.shape[1]):
            if min(abs(eigenvalues[:, i] - eigenvalues[:, 0])) < tolerance:
                pass
            else:
                break
        print('Minimup gap computed with ' + str(i) + 'th eigenvalue')
        gap = eigenvalues[:, i] - eigenvalues[:, 0]

        mingap = min(gap)
        mingap_idx = np.argmin(gap)
        mingap_s = s_plot[mingap_idx]
        mingap_ss.append(mingap_s)
        mingap_index = df.index[df['s'] == mingap_s].tolist()

        print('Minimup gap: ' + str(mingap) + ' GHz')
        print('At s= ' + str(mingap_s))

        idx_end = mingap_index[0] + idx_interval
        idx_start = mingap_index[0] - idx_interval
        index_ss.append(idx_start)
        index_ss.append(idx_end)
        index_ss.append(mingap_index[0])
        idx_interval = int(idx_interval / decrease)
        print(idx_interval)

    print("simulation took {0:.4f} sec".format(time() - ti))
    eigenvals = np.array(eigenvals)
    # np.save(os.path.join("results\mingap\eigenvalues", "four_l" + str(k)), eigenvals)
    # np.save(os.path.join("results\mingap\eigenvalues", "four_idx_l" + str(k)), s_plot)

    draw_plots = True

    if draw_plots:
        df.plot(x='s', y=['A(s) (GHz)', 'B(s) (GHz)'])
        plt.xlabel('Adimensional time $s$')
        plt.ylabel('Energy (GHz)')
        plt.legend()
        plt.xlim(0, 1)
        plt.savefig(r'results\mingap\energies.png')
        #
        plt.figure('all_eigs')
        plt.plot(s_plot, eigenvals)
        plt.xlabel('Adimensional time $s$')
        plt.ylabel('Hamiltonian eigenvalues')
        plt.xlim(0, 1)
        plt.savefig(r'results\mingap\all_eigs_lasserre.png')
        #
        plt.figure()
        plt.plot(s_plot, eigenvals, '0.75')
        plt.plot(s_plot, eigenvals[:, i])
        plt.plot(s_plot, eigenvals[:, 0])
        plt.axvline(mingap_s, color='k', linestyle="dashed")
        plt.xlabel('Adimensional time $s$')
        plt.ylabel('Hamiltonian eigenvalues')
        plt.xlim(0, 1)
        plt.savefig(r'results\mingap\eigs_lasserre.png')

        plt.figure()
        plt.plot(s_plot, gap, '-*')
        plt.vlines(mingap_s, 0, mingap, linestyle="dashed")
        plt.hlines(mingap, 0, mingap_s, linestyle="dashed")
        plt.xlabel('Adimensional time $s$')
        plt.ylabel('Eigenvalues gap $\Delta$')
        plt.xlim(0, 1)
        plt.ylim(0, None)
        plt.savefig(r'results\mingap\gap_lasserre.png')
        # plt.show()
