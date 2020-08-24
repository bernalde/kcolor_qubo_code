*Set size can be set to any value
set i /1*3/;
alias(i,j);

set E(i,j);

loop(i,
	E(i,i ++ 1) = yes;
);
display E;
