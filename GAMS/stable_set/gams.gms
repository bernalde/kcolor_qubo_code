*  MIQCP written by GAMS Convert at 06/03/20 22:05:12
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*          1        1        0        0        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*          7        1        6        0        0        0        0        0
*  FX      0        0        0        0        0        0        0        0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*          7        1        6        0
*
*  Solve m using MIQCP maximizing x7;


Variables  b1,b2,b3,b4,b5,b6,x7;

Binary Variables  b1,b2,b3,b4,b5,b6;

Equations  e1;


e1.. b1 - 7*(sqr((-1) + b1 + b2 + b4) + sqr((-1) + b2 + b3 + b5) + sqr((-1) + 
     b1 + b3 + b6)) + b2 + b3 - x7 =E= 0;

Model m / all /;

m.limrow=0; m.limcol=0;

Solve m using MIQCP maximizing x7;
