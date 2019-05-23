%%% Linear Programming %%%%

%%% minimize 8x1+x2 given 
%%% (1) x1+2x2 >= -14 ---> -x1-2*x2<=14
%%% (2) -4x1-x2<= -33
%%% (3) 2x1+x2<= 20


f=[8;1];
num_var=2;

A=[-1 -2;
    -4 -1;
    2 1];

b=[14;-33;20];

x=intlinprog(f,num_var,A,b)

