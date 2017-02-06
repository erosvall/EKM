%% Dictionary learning algorithm 
noIt  = 10; % sparcit

n = 100; % dimension av data
p = 1000; % antal data
m = 200; % storlek av dictionary

D0 = normc(rand(n,m)); 
D = D0;
X = rand(n,1);
R = X;

%%
%A = zeros(m,p); 
A = [];
S = zeros(m,p);


for i = 1:size(X,2)
    for l = 1:noIt        
        [argval, argmax] = max(abs(X(:,i)'*D)); %Hittar b�sta atomen
        A = [A D(:,argmax)]; % sparar den b�sta atomen
        d = D(:,argmax); % temp atom i detta steg
        D(:,argmax) = []; % Uppdaterar vilka atomer vi anv�nder  
        

        tempInv = inv(d'*d)*d';
        P = d*tempInv;
        %ber�knar ortogonala projektionen
        if(i == 1)
           R(:,i) = (eye(size(P,1))-P)*R(:,i);
           n = norm(R(:,i))
        end
        % sparar koefficienterna f�r sparce-represenationen
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

%% snodd omp algoritm �r b�ttre �n ingen 
[YFIT,R,COEFF,IOPT,QUAL,X] = wmpalg('OMP',X,D,'itermax',10);


