clear all

kernel = 'poly';
kernelparam = 2;

cd /Users/erikrosvall/github/KEX/K-ELM/
addpath('~/Dropbox/Kex/Datasets/Data från Ayman');
%% MNIST

load MNISTData.mat
MNISTacc = [];
MNISTtimeTrain = [];
MNISTtimeClass = [];
datasize = 10000:5000:10000;
for i = datasize
    X = imagesTrain(:,1:i);
    L = labelsTrain(1:i,1)'+1;

    hiddenNodes = size(X,1)*2;
    lambda = 10;
    
    t = cputime();
    [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
    MNISTtimeTrain = [MNISTtimeTrain, cputime - t];
    
    %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
    %trainAccuracy = nnz(l == L)/size(l,2)
    t = cputime();
    testL = KELMclassifier(imagesTest,sigma,wi,wo,kernel,kernelparam);
    MNISTtimeClass  = [MNISTtimeClass, cputime - t];
    
    
    MNISTacc = [MNISTacc, nnz(testL == labelsTest'+1)/size(testL,2)];    
end

clear imagesTrain imagesTest labelsTest labelsTrain sigma testL wi wo X L t
save(strcat('MNIST',kernel));

%% RANDOM FACES AR

load randomfaces4AR.mat

ARacc = [];
ARtimeTrain = [];
ARtimeTrain = [];
datasize = 100:1:100;
for i = datasize
    X = featureMat(:,1:i);
    L = labelMat(:,1:i);

    hiddenNodes = size(X,1)*2;
    lambda = 1;
    
    t = cputime();
    [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
    ARtimeTrain = [ARtimeTrain, cputime - t];
    
    %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
    %trainAccuracy = nnz(l == L)/size(l,2)
    t = cputime();
    testL = KELMclassifier(imagesTest,sigma,wi,wo,kernel,kernelparam);
    ARtimeTrain  = [ARtimeTrain, cputime - t];
    
    
    ARacc = [ARacc, nnz(testL == labelsTest'+1)/size(testL,2)];    
end

clear featureMat filenameMat labelMat
save(strcat('AR',kernel))

%% YALEFACES EXTENDED

%% CALTECH101

%% SCENE15
