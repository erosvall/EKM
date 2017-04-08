function [ inputWeights, outputWeights ] = ELMwithKernelTraining(features, labels, hiddenNodes, lambda,varargin)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    switch nargin
        case 4
            p = 2;
        case 5
            p = varargin{1};
    end
    
    
    % Activation function   
    rlu = @(w,x,b) max(0,w*x+b);
  
    % --- Internal representations ---
    X = features;               % dxN matrix with column vectors of features
    L = labels;                 % The row vector of labels in X
    h = hiddenNodes;            % Number of hidden nodes
    
    % --- Non-user defined variables ---
    N = size(X,2);              % Number of datapoints
    d = size(X,1);              % Dimension of each feature vector. 
    c = size(unique(L),2);      % Number of classes. 
    
    % --- Creation of model matrixes --- 
    b = ones(h,1);              % Initialize bias term
    rng(1337);
    inputWeights = normc(2*rand(h,d)-1);  % Random weights and normalized

    Y = zeros(c,N);             % One-hot label matrix
    for i = 1:N
        Y(L(1,i)+1,i) = 1;
    end
   
    % --- Calculations ---
    sigma = rlu(inputWeights,X,b);        % Applying nonlinear function
    size(sigma)
    KernelMatrix = normc(polyKerl(sigma,sigma,p));
    
    outputWeights = Y*((KernelMatrix+lambda*eye(N))\KernelMatrix'); % Solution to minimizing cost function. 
    
end

