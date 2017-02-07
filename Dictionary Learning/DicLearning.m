%% Dictionary learning algorithm 

clear all
clc

sparcity  = 10; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
m = 200; % storlek av dictionary

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
for i = 1:15
    A = OMP(D,X,sparcity);
    
    D = KSVD(D,X,A);

    F = [F; i norm(D-D0)];
end
G = D0-D;
disp(strcat('Done! Result: ',' ', num2str(norm(G))));
plot(F(:,1),F(:,2))






























