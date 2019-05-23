%%% SCM Linear Programming Model %%%

%%% Objective Function Max F= 2*x1+ 3*x2+ 7*x3
%%% s.t, x1+x2+x3=100
%%% s.t, 2*x1+ 3*x2+ 7*x3 <= 4*100

clc,clear all
myproblem=optimproblem('ObjectiveSense','max');
x=optimvar('x', 3,1,'LowerBound',0,'UpperBound',100);
myproblem.Objective=12*x(1)+5*x(2)+17*x(3);

cons1=x(1)+x(2)+x(3)<=100;
cons2=12*x(1)+5*x(2)+17*x(3)<=3200;

myproblem.Constraints.cons1=cons1;
myproblem.Constraints.cons2=cons2;

showproblem(myproblem)

sol=solve(myproblem);
sol.x

