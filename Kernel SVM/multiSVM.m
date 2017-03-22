function [ res ] = multiSVM( features, labels, kernel, slackPressure, threshold )
%MultiSVM Makes an One-vs-All SVM classifier and returns m SVMs
%
% Input:
%   features - Data to be trained on. Parsed as a NxM matrix. Where N is the
%       number of datapoints and M is the number of datapoints
%   labels - The classes for the data. Given as a Mx1 vector
%   slackPressure - What is the maximum pressure a point may push the
%       decision boundary with. This introduces slack into the model. Default
%       value is 0.
%   threshold - At what point do we consider an datapoint not part of the
%       decisionboundary. Default value is 10^-5
%   kernel - What Kernel function do we want? Accepted input is 
%       > 'lin' which gives a liniear Kernel - Default
%       > 'poly' which gives a polynomial Kernel
%       > 'rad' which gives a radial Kernel
%
%
% Output:
%   res - Mx1 cell array with m classifiers
%
% Dependencies:
%   polyKerl.m
%   linKerl.m
%   radKerl.m 

%% TODO:
% Generalize Kernel input from linKerl
% Handle which input arguments that is required and which ones that has a
% default value.
% Verify output of function with res
%%
if nargin(3) ~= 'lin'
    switch nargin(3)
        case {''}
            %kernel = linkerkl
        case {'poly'}
            %kernel = polyKerl
        case {'rad'}
            % kernel = radKernel
    end
end

% Set up necessary stuff...
n = size(features,1); % Length of data
P = zeros(n,n);
q = -ones(n,1);
h = zeros(2*n,1); % Constraint on optimization criterion Ax =< b
G = -speye(2*n,n);
lbl = labels; % temporary labels holder that we can manipulate for each iteraton.

if nargin(4) == ''
    threshold = 10^(-05);
end

if nargin(5) == ''
    slackPressure = inf;
end

for k = 1:size(unique(labels),1)
    % Label the classes for this instance
    lbl(labels == k) = 1;
    lbl(labels ~= k) = -1;
    % Set up optimization program
    for i = 1:n
        for j = 1:n
            P(i,j) = lbl(i)*lbl(j)*linKerl(features(:,i),features(:,j));
        end
    end
    r = quadprog(P,q,G,h);
    % Find decision boundary for class k.
    
    
    nonZeroAlpha = find(r > threshold & r < slackPressure);
    res = [];
    ii = 1;
    for i = nonZeroAlpha'
        res(k,ii,:) = [r(i), features(:,i)', lbl(i)];
        ii = ii + 1; % loop i rez matrisen
    end
    
    
end

end

