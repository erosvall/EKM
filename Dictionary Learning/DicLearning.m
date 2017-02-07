%% Dictionary learning algorithm 
noIt  = 5; % sparcit

n = 5; % dimension av data
p = 10; % antal data
m = 20; % storlek av dictionary

D0 = normc(rand(n,m)); 
D = D0;
X = rand(n,p);
R = X;
%% OMP funktion
A = OMPfunc(X,D0,noIt);

%% K-SVD

for i = 1:size(D,2) 
    omega = find(A(i,:));
    if(size(omega,2) ~= 0)
        OMEGA = zeros(1,size(omega,2));
        for j = 1:size(omega,2)
           OMEGA(omega(1,j),j) = 1;
        end
        % Multiplicera x^k_T*Omega
        % Multiplicera X*Omega
        % E_k = X-sum(dk*x^k_T)
        % Multiplicera E_k*Omega
        % (1) minimera ||E_k*Omega - d_k*x^k_T*Omega||^2
        % Applicera SVD på (1)
         dk = D(:,i);
    end
end
































