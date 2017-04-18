function [ res ] = radKerl( a,b,sigma )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    asq = sum(a.^2);
    bsq = sum(b.^2);
    
    na = size(a,2);
    nb = size(b,2);
    
%     i = asq'*ones(1,nb);
%     j = ones(na,1)*bsq;
%     temp = 2*a'*b;
    
    D = asq'*ones(1,nb) + ones(na,1)*bsq - 2*a'*b;
 
    res = exp(-D/(2*sigma^2));
    
end

