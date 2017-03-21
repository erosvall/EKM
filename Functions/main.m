
%% ELM on MNIST
clear all
load MNISTData.mat

datasize = 6000;  %max 60000
hiddenLayers = 1000;

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)';
[Wi, Wo] = ELMtrain(X,L,hiddenLayers);

testL = ELMclassifier(imagesTest,Wi,Wo);
accuracy = nnz(testL == labelsTest')/size(testL,2)

%% ELM on YaleFaces
clear all
load yalefaceData.mat

datasize = 135;
hiddenLayers = 1000;

X = yaleFeatures(:,1:datasize);
L = yaleLabels(1,1:datasize);
[Wi, Wo] = ELMtrain(X,L,hiddenLayers);

testL = ELMclassifier(yaleFeatures(:,datasize+1:end),Wi,Wo);
accuracy = nnz(testL == yaleLabels(1,datasize+1:end))/size(testL,2)