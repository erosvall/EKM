function [ totalDiff, diffPerElement ] = DictionaryComparison( D, D0 )
% Comparing similary between original dicionary D0 and generated dictionary
% D. Works only when D and D0 have normalized columns
% 
totalDiff = 0;

for i = 1:size(D,2)
    [maxval, maxarg] = max(abs(D(:,i)'*D0));
    D0(:,maxarg) = [];
    totalDiff = totalDiff + (1-maxval);
end

diffPerElement= totalDiff/size(D,2);
end

