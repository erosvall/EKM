function [ labels ] = KELMclassifier(features,traininfeatures, inputWeights, outputWeights)
% ELMclassifier.m is given the genereated inputweights and the learnt
% outputweights from ELMtrain.m to label the new featurevectors given.  
%   Detailed explanation goes here  
    
    sigma = max(0,inputWeights*features);
    sigmatrain = max(0,inputWeights*traininfeatures);
    
    K = polyKerl(sigmatrain',sigma,2);
    
    Yres = outputWeights * K;
    [~, labels] = max(Yres,[],1);
end

