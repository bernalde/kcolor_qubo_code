$include cycle.inc
*$include devil5.inc
*$include devil50.inc
*$include devil500.inc

binary variable x(i);

binary variable s(i,j);

variable alpha;

equation objective;

parameter M;

M = 2*card(i) + 1;

objective.. sum(i,x(i)) - M*sum(E(i,j),sqr(x(i) + x(j) + s(i,j) - 1)) =E= alpha;

model stableset /all/;

option miqcp = convert;

solve stableset using miqcp max alpha;

display alpha.l;
