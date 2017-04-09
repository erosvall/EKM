function [ inputWeights, outputWeights ] = ELMtrain( features, labels, hiddenNodes,varargin)
    % Labels must be numbered 1-c
    %   Detailed explanation goes here

    % Non-linear activation function used at each hidden node 
    rlu = @(w,x,b) max(0,w*x+b);

    % --- Internal representation ---
    X = features;           
    L = labels;
    h = hiddenNodes;
    
    N = size(X,2);                      % Number of feature vectors
    d = size(X,1);                      % Dimension of a feature vector
    c = size(unique(L),2);              % Number of classes
    
    % --- Initialize matrixes ---
    b = 1;                      % Initialize bias term
    rng(1337)                           % RNG seed for repetability
    inputWeights = normc(2*rand(h,d)-1); 
    
    Y = zeros(c,N);                     % One-hot representation
    for i = 1:N
        Y(L(1,i),i) = 1;
    end
    

    
    % --- Calculations ---
    sigma = rlu(inputWeights,X,b);
    
    switch nargin
        case 4
            sigma = sigma.^varargin{1};
    end
    
    outputWeights = Y*pinv(sigma);      % Output matrix minimizing SSE-cost function 
end

