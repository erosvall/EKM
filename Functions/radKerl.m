function [ res ] = radKerl( a,b,sigma )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%res = exp(-sum((a-b).^2)/(2*sigma^2));
res = zeros(size(a,2),size(b,2));

for i = 1:size(b,2)
    bvec = b(:,i);
 
    res(:,i) = exp(-sum((a-bvec).^2)/(2*sigma^2))';
end

