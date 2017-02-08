%% Dictionary learning algorithm 

clear all
close all
clc

sparcity  = 10; % sparcit

n = 200; % dimension av data
p = 2000; % antal data
m = 200; % storlek av dictionary

D0 = normc(rand(n,m)); 
D =  normc(rand(n,m));

%disp(strcat('Initial difference ',num2str(norm(D0-D))))

A0 = zeros(m,p);
for i = 1:size(A0,2)
    for j = 1:sparcity
       A0(randi(size(A0,1),1),i) = rand(1); 
    end
end

X = D0*A0;
%% D

F = zeros(100,2);
for i = 1:15
    A = OMP(D,X,sparcity);

    %D = KSVD2(D,X,A);
    D = MOD(X,A);
 
    F = [F; i norm(X-D*A)];
end
G = X-D*A;
disp(strcat('Done! Result: norm(X-X_hat)=', num2str(norm(G))));
plot(F(:,1),F(:,2))

% Dictionary comparison
Dcopy = D0;
totalDiff = 0;

for i = 1:size(D:2)
    temp = D(:,i) - Dcopy;
    [minval, minarg] = min(sqrt(sum(temp.^2)));
    Dcopy(:,minarg) = [];
    totalDiff = totalDiff + minval;
end




























