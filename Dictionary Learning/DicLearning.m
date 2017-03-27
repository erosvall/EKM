%% Dictionary learning algorithm 

clear all
close all
clc

sparcity  = 10; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
m = 200; % storlek av dictionary

D0 = dictmake(n,m,'U');
D =  dictmake(n,m,'U');

disp(strcat('Initial difference ',num2str(DictionaryComparison(D,D0))))

A0 = zeros(m,p);
for i = 1:size(A0,2)
    for j = 1:sparcity
       A0(randi(size(A0,1),1),i) = rand(1); 
    end
end

X = D0*A0;

w = warning('on','all');
P1 = [];
A = zeros(m,p);
P2 = [];
%P1 = [P1; 0 DictionaryComparison(D,D0)];

for i = 1:1

    A = OMP(D,X,sparcity);
    %D = MOD(X,A);
    D = SVDDictionaryUpdate(D,X,A);
    [totalDiff, diffPerCol] = DictionaryComparison(D,D0);
    P1 = [P1; i diffPerCol];
end
figure(1)
plot(P1(:,1),P1(:,2))

disp(strcat('Done! Result: DicComp(D-D0)=', num2str(DictionaryComparison(D,D0))));



























