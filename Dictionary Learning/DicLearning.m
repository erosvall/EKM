%% Dictionary learning algorithm 
noIt  = 5; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
m = 200; % storlek av dictionary

D0 = rand(n,m); 
D = D0;
X = rand(n,p);

%%
A = zeros(m,p); 


for i = 1:size(X,2)
    for l = 1:noIt        
        [temp, argmax] = max(abs(X(:,i)'*normc(D)));
        argval = X(:,i)'*normc(D(:,argmax));
        A(argmax,i) = argval;
        X(:,i) = X(:,i) - D(:,argmax)*argval;
        
    end
end
