function [ mu, sigma ] = GaussianTrain( features, labels )
%UNTITLED3 Summary of this function goes here
%   Every datapoint has its features sorted as a columvector of the 
%   "features" matrix. Corresponding lables are given as a row-vector in 
%   "labels".

% Usefull information for preallocating matrixes 
numbOfClasses = size(unique(labels),2);
dimOfData = size(features,1);

% Preallocating 
mu = zeros(dimOfData,numbOfClasses);
sigma = zeros(dimOfData,dimOfData,numbOfClasses);

% Learningprocces
for i = 1:numbOfClasses 
    mu(:,i) = mean(features(:,labels == i),2);
    sigma(:,:,i) = cov(features(:,labels == i)')+10e-3*eye(dimOfData);
end

return

