function [ labels ] = ELMclassifier(features, inputWeights, outputWeights,varargin)
% ELMclassifier.m is given the genereated inputweights and the learnt
% outputweights from ELMtrain.m to label the new featurevectors given.  
%   Detailed explanation goes here

    rlu = @(w,x,b) max(0,w*x+b);
    sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));
    
    Btest = ones(size(inputWeights,1),size(features,2));
    
    switch nargin
        case 4
            switch varargin{1}                            
                case 'rlu'
                    sigmatest = rlu(inputWeights,features,Btest); 
                case 'sigmoid'
                    sigmatest = sigmoid(inputWeights,features,Btest);
            end
        case 3
            sigmatest = rlu(inputWeights,features,Btest); 
    end
   
    Yres = outputWeights * sigmatest;
    [~, maxind] = max(Yres,[],1);
    labels = maxind-1;
    
return

