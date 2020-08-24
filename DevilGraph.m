function[Adjacency, Edges, NoNodes, Alpha] = DevilGraph(n)
%
% INPUT:
% n = {0,1,2,3,...} rank of the graph (the higher the bigger the graph)
% OUTPUT:
% Adjacency = node to node adjacency matrix
% Edges = Edges of the Graph
% NoNodes = Number of nodes of the Graph
% Alpha = Its stable set number

    Jn = ones(n,n);
	In = eye(n);
	en = ones(n,1);
	zn = zeros(n,1);
	Zn = zeros(n,n);
	A = [0  1  en'   zn'   zn'; 
     1  0  zn'   en'   zn'; 
	 en zn Zn    Jn-In In; 
	 zn en Jn-In Zn    In; 
	 zn zn In    In    Zn];
 
Adjacency = A; 
 
[I, J] = find(tril(A)==1);
Edges = [I';J'];

NoNodes = length(A);

Alpha = n+1;