%% Dictionary learning algorithm 

clear all
clc

sparcity  = 4; % sparcit

n = 40; % dimension av data
p = 300; % antal data
m = 100; % storlek av dictionary

D0 = normc(rand(n,m)); 
D =  normc(rand(n,m));

A0 = zeros(m,p);
for i = 1:size(A0,2)
    for j = 1:sparcity
       A0(randi(size(A0,1),1),i) = rand(1); 
    end
end

X = D0*A0;
%% D

F = [];
for i = 1:100
    A = OMP(D,X,sparcity);
    D = KSVD(D,X,A);
    F = [F; i norm(D-D0)];
end
G = D-D0;
disp(strcat('Done! Result: ', num2str(norm(G))));
plot(F(1,:),F(2,:))






























