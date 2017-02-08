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

disp(strcat('Initial difference ',num2str(DictionaryComparison(D,D0))))

A0 = zeros(m,p);
for i = 1:size(A0,2)
    for j = 1:sparcity
       A0(randi(size(A0,1),1),i) = rand(1); 
    end
end

X = D0*A0;
%% D
w = warning('off','all');
P1 = [];
P2 = [];
for i = 1:30
    A = OMP(D,X,sparcity);
    D = MOD(X,A);
    %D = KSVD2(D,X,A);
    [totalDiff, diffPerCol] = DictionaryComparison(D,D0);
    
    P1 = [P1; i totalDiff];
    
    
    
end

disp(strcat('Done! Result: norm(D-D0)=', num2str(norm(D-D0))));
figure(1)
plot(P1(:,1),P1(:,2))


disp(strcat('Done! Result: DicComp(D-D0)=', num2str(DictionaryComparison(D,D0))));



























