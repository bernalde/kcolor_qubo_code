$include cycle.inc
*$include devil5.inc
*$include devil50.inc
*$include devil500.inc

binary variable x(i);

variable alpha;

equation objective;

objective.. sum(i,x(i)) =E= alpha;

equation constraint;

constraint(i,j)$E(i,j).. x(i)*x(j) =E= 0;

model stableset /all/;

option miqcp = gurobi;

solve stableset using miqcp max alpha;

display alpha.l;
