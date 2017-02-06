%% Dictionary learning algorithm 
noIt  = 10; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
m = 200; % storlek av dictionary

D0 = normc(rand(n,m)); 
D = D0;
X = rand(n,p);
R = X;

%%
%A = zeros(m,p); 
A = [];
S = zeros(m,p);


for i = 1:size(X,2)
    for l = 1:noIt        
        [argval, argmax] = max(abs(X(:,i)'*D)); %Hittar bästa atomen
        A = [A D(:,argmax)]; % sparar den bästa atomen
        d = D(:,argmax); % temp atom i detta steg
        D(:,argmax) = []; % Uppdaterar vilka atomer vi använder  
        

        tempInv = inv(d'*d)*d';
        P = d*tempInv;
        %beräknar ortogonala projektionen
        if(i == 1)
           R(:,i) = (eye(size(P,1))-P)*R(:,i);
           n = norm(R(:,i))
        end
        % sparar koefficienterna för sparce-represenationen
        S(argmax,i) = tempInv*X(:,i);
        
 
    end
end
%%
S = [];
R = X;
for i = 1:size(X,2)
    for l = 1:noIt
        [argval, argmax] = max(abs(R(:,i)'*D));
        R(:,i) = R(:,i) - D(:,argmax)*argval;
        norm(R(:,i))
        S(argmax,i) = argval;
        D(:,argmax) = [];
    end
    
end

%% snodd omp algoritm är bättre än ingen 
S = zeros(m,p);
[YFIT,R,COEFF,IOPT,QUAL,L] = wmpalg('OMP', X, D,'itermax',noIt);
for i = 1:size(COEFF,1)
    S(IOPT(1,i),1) = COEFF(i,1);
end

%% K-SVD

for k = 1:size(D,2) 
    omega = find(S(k,:));
    if(size(omega,2) ~= 0)
        OMEGA = zeros(1,size(omega,2));
        for i = 1:size(omega,2)
           OMEGA(omega(1,i),i) = 1;
        end
        % Multiplicera x^k_T*Omega
        % Multiplicera X*Omega
        % E_k = X-sum(dk*x^k_T)
        % Multiplicera E_k*Omega
        % (1) minimera ||E_k*Omega - d_k*x^k_T*Omega||^2
        % Applicera SVD på (1)
         dk = D(:,k);
    end
end






























