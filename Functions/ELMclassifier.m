function [ labels ] = ELMclassifier(features, inputWeights, outputWeights)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

rlu = @(w,x,b) max(0,w*x+b);
sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));

Btest = ones(size(inputWeights,1),size(features,2));

sigmatest = sigmoid(inputWeights,features,Btest);

Yres = outputWeights * sigmatest;
[~, maxind] = max(Yres,[],1);
labels = maxind-1;

return
