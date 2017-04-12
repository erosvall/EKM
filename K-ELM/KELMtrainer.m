function [ inputWeights, outputWeights ] = KELM(features, labels, hiddenNodes,lambda)
    % Labels must be numbered 1-c
    %   Detailed explanation goes here

    % --- Internal representation ---
    X = features;           
    L = labels;
    h = hiddenNodes;
    
    N = size(X,2);                      % Number of feature vectors
    d = size(X,1);                      % Dimension of a feature vector
    c = size(unique(L),2);              % Number of classes
    
    % --- Initialize matrixes ---   
    rng(1337)                           % RNG seed for repetability
    inputWeights = normc(2*rand(h,d)-1); 
        
    % --- Calculations ---
    Y = zeros(c,N);                     % One-hot representation
    for i = 1:N
        Y(L(1,i),i) = 1;
    end

    sigma = max(0,inputWeights*X);      % RLU activation 
    
    K = polyKerl(sigma',sigma,2);
    
    
    outputWeights = (Y/(K'*K+lambda*eye(N)))*K';      % Output matrix minimizing SSE-cost function 
end
