%% Data Generation for K-means

function [ Data ] = NormalDataGeneration( numberOfClusters,sizeOfCluster,dimensionOfData )
%Generates culsters of normal distributed data in 2d for k-means testing
n  = numberOfClusters;
s = sizeOfCluster;
d = dimensionOfData;
Data = zeros(n*s,d);

for i = 1:n
    mu = rand();
    sigma = abs(rand())/10;
    Data((i-1)*s+1:(i-1)*s+1+s,:) = normrnd(mu,sigma,s,d);

end