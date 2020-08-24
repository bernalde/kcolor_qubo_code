*$include cycle.inc
*$include devil5.inc
$include devil50.inc
*$include devil500.inc

set r /1*3/;
alias(r,rr);

binary variable x(i,r);

variable alpha;

equation objective;

objective.. sum((i,r),x(i,r)) =E= alpha;

equation constraint1, constraint2;

constraint1(i,j,r)$E(i,j).. x(i,r)*x(j,r) =E= 0;

constraint2(i,r,rr)$(ord(r) ne ord(rr)).. x(i,r)*x(i,rr) =E= 0;

model coloring /all/;

option miqcp = gurobi;

solve coloring using miqcp max alpha;

display alpha.l;
