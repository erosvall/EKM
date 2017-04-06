function [ labels ] = ELMwithKernelClassifier(features, inputWeights, outputWeights,NonLinearFunction)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    rlu = @(w,x,b) max(0,w*x+b);
    sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));
    
    Btest = ones(size(inputWeights,1),1);
    
    switch NonLinearFunction
        case 'rlu'
            sigmatest = rlu(inputWeights,features,Btest); 
        case 'sigmoid'
            sigmatest = sigmoid(inputWeights,features,Btest);
        otherwise
            sigmatest = rlu(inputWeights,features,Btest); 
    end
    
    %sigmatest = polyKerl(sigmatest',sigmatest,2);
    
    Yres = outputWeights * sigmatest.^3 ;
    [~, maxind] = max(Yres,[],1);
    labels = maxind-1;
    
return
