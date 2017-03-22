
%% ELM on MNIST 60 000 training, 10 000 testing
clear all
load MNISTData.mat

datasize = 6000;  %max 60000
hiddenLayers = 1000;

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)';
[Wi, Wo] = ELMtrain(X,L,hiddenLayers);

testL = ELMclassifier(imagesTest,Wi,Wo);
disp('Accurracy on MNIST')
accuracy = nnz(testL == labelsTest')/size(testL,2)

%% ELM on YaleFaces 165 pictures from 15 classes. 
clear all
load yalefaceData.mat

datasize = 135;
hiddenLayers = 1000;

X = yaleFeatures(:,1:datasize);
L = yaleLabels(1,1:datasize);
[Wi, Wo] = ELMtrain(X,L,hiddenLayers);

testL = ELMclassifier(yaleFeatures(:,datasize+1:end),Wi,Wo);
disp('Accurracy on yalefaces')
accuracy = nnz(testL == yaleLabels(1,datasize+1:end))/size(testL,2)

%% GaussianClassifier on MNIST
clear all
load MNISTData.mat

datasize = 60000;

features = imagesTrain(:,1:datasize);
labels = labelsTrain(1:datasize,1)'+1; % numrerade från 0-9 

[mu,sigma] = GaussianTrain(features,labels);

testL = GaussianClassifier(imagesTest,mu,sigma);

accuracy = nnz(testL == labelsTest'+1)/size(testL,2)

