
%% ELM on MNIST 60 000 training, 10 000 testing
clear all
load MNISTData.mat

datasize = 60000;  

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)';

hiddenLayers = 100:100:3000;
error = [];


for i = hiddenLayers
    [Wi, Wo] = ELMtrain(X,L,i);
    testL = ELMclassifier(imagesTest,Wi,Wo);
    error = [error 1-(nnz(testL == labelsTest')/size(testL,2))];
end

scatter(hiddenLayers,error)

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
error = 1 -nnz(testL == yaleLabels(1,datasize+1:end))/size(testL,2)

%% GaussianClassifier on MNIST
clear all
load MNISTData.mat

datasize = 60000;

features = imagesTrain(:,1:datasize);
labels = labelsTrain(1:datasize,1)'+1; % numrerade från 0-9 

[mu,sigma] = GaussianTrain(features,labels);

testL = GaussianClassifier(imagesTest,mu,sigma);

error = 1 - nnz(testL == labelsTest'+1)/size(testL,2)

%% CIFAR 10 dataset
clear all
load '/Users/Viktor/Dropbox/KTH/År 3/Period 4/Kex/Datasets/cifar-10-batches-mat/data_batch_1.mat'


load '/Users/Viktor/Dropbox/KTH/År 3/Period 4/Kex/Datasets/cifar-10-batches-mat/test_batch.mat'

%% Multi SVM
clear all
load MNISTData.mat

datasize = 1000;

features = imagesTrain(:,1:datasize);
labels = labelsTrain(1:datasize,1)'+1;

res =  multiSVM(features,labels,'lin',Inf,10e-5);
testL = classifySVM(imagesTest(:,1123),res,'lin')






