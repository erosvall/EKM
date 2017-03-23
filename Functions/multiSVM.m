function [ res ] = multiSVM( features, labels, kernel, slackPressure, threshold, p )
% MultiSVM Makes an One-vs-All SVM classifier and returns m SVMs
%
% Input:
%   features - Data to be trained on. Parsed as a NxM matrix. Where N is the
%       number of datapoints and M is the number of datapoints
%   labels - The classes for the data. Given as a Mx1 vector
%   kernel - What Kernel function do we want? Accepted input is 
%       > 'lin' which gives a liniear Kernel - Default
%       > 'poly' which gives a polynomial Kernel
%       > 'rad' which gives a radial Kernel
%   slackPressure - What is the maximum pressure a point may push the
%       decision boundary with. This introduces slack into the model. D
%   threshold - At what point do we consider an datapoint not part of the
%       decisionboundary.
%   p
%       Kernel Parameter, contextual. If kernel is radial it's sigma. 
%       If the kernel is polynomial it's p.
%
%
%
%
% Output:
%   res - Mx1 cell array with m classifiers
%
% Dependencies:
%   polyKerl.m
%   linKerl.m
%   radKerl.m 



% Set up necessary stuff...
n = size(features,2); % numberof of data
P = zeros(n,n);
q = -ones(n,1);
h = zeros(2*n,1); % Constraint on optimization criterion Ax =< b
G = -speye(2*n,n);
labels = labels';
lbl = labels; % temporary labels holder that we can manipulate for each iteraton.
res = [];


for k = 1:size(unique(labels),1)
    
    % Label the classes for this instance
    lbl(labels == k) = 1;
    lbl(labels ~= k) = -1;
    
    % Set up optimization program
    switch kernel
        case{'lin'}
            for i = 1:n
                for j = 1:n
                    P(i,j) = lbl(i)*lbl(j)*linKerl(features(:,i)',features(:,j));
                end
            end            
        case{'rad'}
            for i = 1:n
                for j = 1:n
                    P(i,j) = lbl(i)*lbl(j)*radKerl(features(:,i),features(:,j),p);
                end
            end              
        case{'pol'}
            for i = 1:n
                for j = 1:n
                    P(i,j) = lbl(i)*lbl(j)*polyKerl(features(:,i)',features(:,j),p);
                end
            end
        otherwise
            disp('Failed to recognize Kernel function command, defaulting to linear kernel')
            for i = 1:n
                for j = 1:n
                    P(i,j) = lbl(i)*lbl(j)*linKerl(features(:,i)',features(:,j));
                end
            end                
    end
    % Do the heavy lifting
    r = quadprog(P,q,G,h);

    % Find decision boundary for class k.
    nonZeroAlpha = find(r > threshold & r < slackPressure);
    ii = 1;
    for i = nonZeroAlpha'
        res(k,ii,:) = [r(i), features(:,i)', lbl(i)];
        ii = ii + 1; % loop i rez matrisen
    end
    
    
end

end

