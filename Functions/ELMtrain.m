function [ inputWeights, outputWeights ] = ELMtrain( features, labels, hiddenLayers)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Activation Functions which might want to be used.   
rlu = @(w,x,b) max(0,w*x+b);
sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));

X = features;
L = labels;
h = hiddenLayers;
N = size(X,2);                      % Number of datapoints

B = ones(h,size(X,2));              % Initialize bias term
rng(1337)
Wi = normc(2*rand(h,size(X,1))-1);         % random weights [-1,1]

sigma = rlu(Wi,X,B);            % Calculating hiden layer response using sigmoid 
         
Y = zeros(size(unique(L),2),size(X,2));            % Initialize class matrix

% Fill class matrix
for i = 1:size(L,2)
    Y(L(1,i)+1,i) = 1;
end

Wo = Y*pinv(sigma);                 % Calculated weiths in output layer

inputWeights = Wi;
outputWeights = Wo;
return

