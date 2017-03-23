function [ res ] = radKerl( a,b,sigma )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

res = exp(-sum((a-b).^2)/(2*sigma^2));

end

