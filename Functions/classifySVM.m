function [ res ] = classifySVM( newV, alphas, arg )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

class = zeros(size(alphas,1),1);
memory = zeros(size(alphas,1),1);
for j = 1:size(alphas,1)
    switch arg
        case {'lin'}
            v = 0;
            for i = 1:size(alphas,2)
                v = v + alphas(j,i,1)*alphas(j,i,end)*linKerl(newV', reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ) ) ;                
            end
            memory(j) = v;
            class(j) = sign(v);
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
end

if size( find(class == 1) , 2) == 1
    res = find(class == 1);
else
    [~, res] = max(memory);
end

end
