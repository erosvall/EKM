function [ inputWeights, outputWeights ] = ELMtrain( features, labels, hiddenNodes,lambda)
    % Labels must be numbered 1,...,c
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
    
    if size(labels,1)==1
        Y = zeros(c,N);                     % One-hot representation
        for i = 1:N
            Y(L(1,i),i) = 1;
        end
    end
    if size(labels,1) > 1
        Y = labels;
    end
    
    % --- Calculations ---
    sigma = max(0,inputWeights*X);      % RLU on hidden neuron input    
    outputWeights = (Y/(sigma'*sigma+lambda*eye(N)))*sigma';      % Output matrix minimizing SSE-cost function 
end

