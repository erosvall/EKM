function [ v ] = classifySVM( newV, alphas )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

v = 0;
for i = 1:size(alphas,1)
    v = v + alphas(i,1)*alphas(i,4)*linKerl(newV, alphas(i,2:3));

end

