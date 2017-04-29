function [ inputWeights, outputWeights, sigma] = KELMtrainer(features, labels, hiddenNodes,lambda,kernel,varargin)
    % Labels must be numbered 1-c
    %   Detailed explanation goes here

    % --- Internal representation ---
    N = size(features,2);                      % Number of feature vectors
    d = size(features,1);                      % Dimension of a feature vector
    c = size(unique(labels),2);                 % Number of classes
    
    % --- Initialize matrixes ---   
    rng(1337)                           % RNG seed for repetability
    inputWeights = normc(2*rand(hiddenNodes,d)-1); 
        
    % --- Calculations ---
    if size(labels,1) == 1
        Y = zeros(c,N);                     % One-hot representation
        for i = 1:N
            Y(labels(1,i),i) = 1;
        end
    end    
    if size(labels,1) > 1
        Y = labels;
    end
    
    sigma = max(0,inputWeights*features);      % RLU activation 
    
    switch kernel
        case 'poly'
            K = polyKerl(sigma',sigma,varargin{1});
        case 'rbf'
            K = radKerl(sigma,sigma,varargin{1});
    end
    
    if nargin == 7
        disp(strcat('Biggest value of K^T * K: ', num2str(max(max(K'*K)))));
        disp(strcat('Smallest value of K^T * K: ', num2str(min(min(K'*K)))));

    end
    
    outputWeights = (Y/(K'*K+lambda*eye(N)))*K';      % Output matrix minimizing SSE-cost function 
end
