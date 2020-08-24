function[Adjacency, Edges, Alpha] = Cycles(n)
%
% INPUT:
% n = {2,3,...} number of nodes in graph
% OUTPUT:
% Adjacency = node to node adjacency matrix
% Edges = Edges of the Graph
% Alpha = Its stable set number

A = full(gallery('tridiag',n,1,0,1));
A(n,1) = 1;
A(1,n) = 1;

Adjacency = A;

[I, J] = find(tril(A)==1);
Edges = [I';J'];

Alpha = floor(n/2);