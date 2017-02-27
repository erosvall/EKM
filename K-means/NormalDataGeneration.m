%% Data Generation for K-means

function [ Data ] = NormalDataGeneration( numberOfClusters,sizeOfCluster,dimensionOfData )
%Generates culsters of normal distributed data in 2d for k-means testing
n  = numberOfClusters;
s = sizeOfCluster;
d = dimensionOfData;
Data = [];

for i = 1:n
    mu = rand(1,d)*20;
    A = rand(d,d);
    sigma = round((tril(A) + tril(A)'),3);
    
    
    Data = [Data; mvnrnd(mu,sigma,s)];
end
