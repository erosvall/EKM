function [ inputWeights, outputWeights ] = ELMwithKernelTraining(features, labels, hiddenLayers,lambda)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    % Activation Functions which might want to be used.   
    rlu = @(w,x,b) max(0,w*x+b);

    X = features;
    L = labels;
    h = hiddenLayers;
    N = size(X,2);                  % Number of datapoints

    b = ones(h,1);                  % Initialize bias term

    rng(1337);
    Wi = normc(2*rand(h,size(X,1))-1);   % random weights and normalized
    Y = zeros(size(unique(L),2),N); 
    
    % Fill class matrix
    for i = 1:size(L,2)
        Y(L(1,i)+1,i) = 1;
    end
    
    sigma = rlu(Wi,X,b); %applying nonlinear function
    %%sigma = polyKerl(sigma',sigma,2); % applying polynomial kernel
    
    Wo = Y*pinv(sigma.^3);

    inputWeights = Wi;
    outputWeights = Wo;
end
