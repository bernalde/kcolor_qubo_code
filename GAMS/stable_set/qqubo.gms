$include cycle.inc
*$include devil5.inc
*$include devil50.inc
*$include devil500.inc

binary variable x(i);

variable alpha;

equation objective;

parameter M /2/;

objective.. sum(i,x(i)*x(i)) - M*sum(E(i,j),x(i)*x(j)) =E= alpha;

model stableset /all/;

option miqcp = gurobi;

solve stableset using miqcp max alpha;

display alpha.l;
