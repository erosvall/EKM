function [ labels ] = KELMclassifier(features,traininfeatures, inputWeights, outputWeights,kernel,varargin)
% ELMclassifier.m is given the genereated inputweights and the learnt
% outputweights from ELMtrain.m to label the new featurevectors given.  
%   Detailed explanation goes here  
    
    sigma = max(0,inputWeights*features);
    sigmatrain = max(0,inputWeights*traininfeatures);
    
    switch kernel
        case 'poly'
            K = polyKerl(sigmatrain',sigma,varargin{1});
        case 'rbf'
            K = radKerl(sigmatrain,sigma,varargin{1});
    end
    
    Yres = outputWeights * K;
    [~, labels] = max(Yres,[],1);
end

