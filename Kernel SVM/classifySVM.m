function [ v ] = classifySVM( newV, alphas, arg )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


switch arg
    case {'lin'}
        v = 0;
        for i = 1:size(alphas,1)
            v = v + alphas(i,1)*alphas(i,4)*linKerl(newV, alphas(i,2:3));

        end
    case {'rad'}
        v = 0;
        for i = 1:size(alphas,1)
            v = v + alphas(i,1)*alphas(i,4)*polKerl(newV, alphas(i,2:3),2);

        end    
    case {'pol'}
        v = 0;
        for i = 1:size(alphas,1)
            v = v + alphas(i,1)*alphas(i,4)*radKerl(newV, alphas(i,2:3),20);

        end        
end
