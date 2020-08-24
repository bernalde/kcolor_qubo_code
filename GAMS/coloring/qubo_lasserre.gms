*$include cycle.inc
*$include devil5.inc
$include devil50.inc
*$include devil500.inc

set r /1*3/;
alias(r,rr);

binary variable x(i,r);

binary variable s(i,j,r);

binary variable t(i);

variable alpha;

equation objective;

parameter
C1 /1/,
C2 /1/
;

objective.. sum((i,r),x(i,r)) - C1*sum((E(i,j),r),sqr(x(i,r)+x(j,r)+s(i,j,r)-1)) - C2*sum(i,sqr(sum(r,x(i,r)) + t(i) -1)) =E= alpha;

model coloring /all/;

option miqcp = gurobi;

solve coloring using miqcp max alpha;

display alpha.l;
