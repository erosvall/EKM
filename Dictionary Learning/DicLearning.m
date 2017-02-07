%% Dictionary learning algorithm 
sparcity  = 5; % sparcit

n = 5; % dimension av data
p = 7; % antal data
m = 20; % storlek av dictionary

D0 = normc(rand(n,m)); 
D =  normc(rand(n,m));

A0 = zeros(m,p);
for i = 1:size(A,2)
    for j = 1:sparcity
       A(randi(size(A,1),1),i) = rand(1); 
    end
end

X = D0*A0;
%%


for i = 1:5
    disp(i)
    A = OMPfunc(X,D,sparcity);
    size(A)
    D = KSVD(D,X,A);
    
end

D0-D






























