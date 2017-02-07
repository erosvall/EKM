function [ D ] = KSVD( dictionary, signal, sparseRepresentation )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

D = dictionary;
X = signal;
A = sparseRepresentation;

[n,p] = size(signal);


for i = 1:size(D,2) 
    a = A(i,:);
    omega = find(A(i,:));
    v = size(omega,2);
    OMEGA = zeros(p,v);
    if v ~= 0
        for j = 1:v
           OMEGA(omega(1,j),j) = 1;
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

