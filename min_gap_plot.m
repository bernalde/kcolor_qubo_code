N = 25; % Total number of cycle graphs
m = 6; % Number of different reformulation
% m = 1: Proposed reformulation with M=1
% m = 2: Proposed reformulation with M=1 and full diagonal (no linear terms
% in obj)
% m = 3: Proposed reformulation with M=2
% m = 4: Proposed reformulation with M=1 and full diagonal (no linear terms
% in obj)
% m = 5: Lassere reformulation
% m = 6: Lassere reformulation and full diagonal (no linear terms in obj)

mingap = zeros(N,m);
eigenvals = cell(N,m);

for n=3:N
    % Generate graph
%     [A, Edges, Alpha] = Cycles(n);
    [A, Edges, k, Alpha] = DevilGraph(n);
    % First reformulation
    eigen = sort(eig(A));
    eigenvals{n,1} = eigen;
    mingap(n,1) = eigen(2) - eigen(1);
    
    % Second reformulation
    Ad = A + eye(size(A));
    eigen = sort(eig(Ad));
    eigenvals{n,2} = eigen;
    mingap(n,2) = eigen(2) - eigen(1);
    
    % Third reformulation
    A2 = 2*A;
    eigen = sort(eig(A2));
    eigenvals{n,3} = eigen;
    mingap(n,3) = eigen(2) - eigen(1);
    
    % Fourth reformulation
    A2d = A2 + eye(size(A));
    eigen = sort(eig(A2d));
    eigenvals{n,4} = eigen;
    mingap(n,4) = eigen(2) - eigen(1);
    
    % Fifth reformulation
    B = adj2inc(sparse(logical(A)),1);
    y = 2*length(A) + 1;
    B = double(full(B));
    mat = [B'*B, B' ;
        B eye(length(B))];
    eigen = sort(eig(mat));
    eigenvals{n,5} = eigen;
    mingap(n,5) = eigen(2) - eigen(1);
    
    % Sixth reformulation
    e1 = ones(length(B),1); % Ones of size |Edges|
    e2 = ones(length(A),1); % Ones of size |Nodes|
    C = diag(e2-2*y*(B'*e1));
    matd = [B'*B + C, B';
        B (1-2*y)*eye(length(B))];
    eigen = sort(eig(matd));
    eigenvals{n,6} = eigen;
    mingap(n,6) = eigen(2) - eigen(1);
end
set(0,'DefaultAxesLineStyleOrder',{'-+','-o','-*','-.','-x','-s','-d','-^','-v','->','-<','-p','-h'});

figure(1)
% % Plot all lines
% plot(mingap)
% legend({'M=1','M=1 diag','M=2','M=2 diag','Lasserre','Lasserre diag'},'Location','best')

% Plot only non-diagonal reformulations
plot(mingap(:,[1,3,5]))
legend({'M=1','M=2','Lasserre'},'Location','best')

ylim([-1,inf])
% xlabel('Number of nodes n')
xlabel('Rank of graph')
ylabel('Minimum gap \Delta')
% title('Cycle graphs minimum gap agains size different reformulations')
title('Devil graphs minimum gap agains size different reformulations')