function [ labels ] = GaussianClassifier( features, mu, sigma )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

% Size data for preallocating and looping. 
numbOfClasses = size(mu,2);
numbOfData = size(features,2);

% Preallocating labelsvector
%labels = zeros(1,numbOfData);
probability = zeros(numbOfData,numbOfClasses);

% Compairing all featurvectors to one classdistribution at a time
% and assigning label as the class with highest likelihood of producing 
% this featurevector. 

for i = 1:numbOfClasses %looping through the model
    probability(:,i) = mvnpdf(features', mu(:,i)', sigma(:,:,i));    
end
[~,labels] = max(probability,[],2);
labels = labels';

return

