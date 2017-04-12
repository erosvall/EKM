function [ labels ] = ELMwithKernelClassifier(features, inputWeights, outputWeights,varargin)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    
    rlu = @(w,x,b) max(0,w*x+b);    %Non-linear activation function
    
    h = size(inputWeights,1);       % Number of hidden nodes 
    
    switch nargin
        case 4
            p = 1;                  % Default kernel degree
        case 5
            p = varargin{1};
    end
    
    b = ones(h,1);

    sigma = rlu(inputWeights,features,b); 

    KernelMatrix = normc(polyKerl(sigma,sigma',p));
    
    Yres = outputWeights * KernelMatrix;
    [~, labels] = max(Yres,[],1);
end

