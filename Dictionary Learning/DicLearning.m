%% Dictionary learning algorithm 

clear all
close all
clc

sparcity  = 10; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
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

F = [];
for i = 1:100
    A = OMP(D,X,sparcity);

    D = KSVD2(D,X,A);
 
    F = [F; i norm(X-D*A)];
end
G = X-D*A;
disp(strcat('Done! Result: norm(X-X_hat)  ',' ', num2str(norm(G))));
plot(F(:,1),F(:,2))






























