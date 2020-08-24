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

constraint1(i,j,r)$E(i,j).. x(i,r) + x(j,r) =L= 1;

*constraint2(i)$(ord(r) ne ord(rr)).. x(i,r) + x(i,rr) =L= 1;
constraint2(i).. sum(r,x(i,r)) =L= 1;

model coloring /all/;

option mip = gurobi;

solve coloring using mip max alpha;

display alpha.l;
