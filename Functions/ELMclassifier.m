function [ labels ] = ELMclassifier(features, inputWeights, outputWeights,varargin)
% ELMclassifier.m is given the genereated inputweights and the learnt
% outputweights from ELMtrain.m to label the new featurevectors given.  
%   Detailed explanation goes here

    rlu = @(w,x,b) max(0,w*x+b);
    sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));
    
    b = ones(size(inputWeights,1),size(features,2));
    
    switch nargin
        nargin
        case 3
            sigma = rlu(inputWeights,features,b); 
        case 4
            switch varargin{1}                            
                case 'rlu'
                    sigma = rlu(inputWeights,features,b); 
                case 'sigmoid'
                    sigma = sigmoid(inputWeights,features,b);
            end
        case 5
            sigma = sigma.^(varargin{2});
    end
    
    
    Yres = outputWeights * sigma;
    [~, labels] = max(Yres,[],1);
return

