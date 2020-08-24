*$include cycle.inc
*$include devil5.inc
$include devil50.inc
*$include devil500.inc

set r /1*3/;
alias(r,rr);

binary variable x(i,r);

variable alpha;

equation objective;

parameter
C1 /2/,
C2 /2/
;

objective.. sum((i,r),x(i,r)) - C1*sum((E(i,j),r),x(i,r)*x(j,r)) - C2*sum((i,r,rr)$(ord(r) ne ord(rr)),x(i,r)*x(i,rr)) =E= alpha;

model coloring /all/;

option miqcp = gurobi;

solve coloring using miqcp max alpha;

display alpha.l;
