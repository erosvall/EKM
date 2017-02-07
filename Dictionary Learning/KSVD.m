function [ D ] = KSVD( dictionary, signal, sparseRepresentation )
%KSVD Returns an optimized dictionary from intial input
%   Detailed explanation goes here

D = dictionary;
X = signal;
A = sparseRepresentation;

[n,p] = size(signal);


for i = 1:size(D,2) 
    omega = find(A(i,:)); % Hitta alla index i A som används av atom i
    v = size(omega,2); % Enbart hjälp
    OMEGA = zeros(p,v);
    if v ~= 0
        for j = 1:v
           OMEGA(omega(1,j),j) = 1; % Matrisen som komprimerar A till enbart aktiva rader
        end
    end
    I = zeros(n,p);
    for k = 1:size(D,2)
       if k ~= i
           d_k = D(:,k);
           a_k = A(k,:);
           I = I + d_k * a_k;
       end
    end
    Ei = X - I;
    Er = Ei*OMEGA;
    [U,S,V] = svd(Er);
    D(:,i) = U(:,1);

end

end

