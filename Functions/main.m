
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
%A = load('/Users/Viktor/Dropbox/KTH/År 3/Period 4/Kex/Datasets/cifar-10-batches-mat/data_batch_1.mat');
%B = load('/Users/Viktor/Dropbox/KTH/År 3/Period 4/Kex/Datasets/cifar-10-batches-mat/test_batch.mat');
[X1,Y1,y1] = LoadBatch('data_batch_1.mat');
[X2,Y2,y2] = LoadBatch('data_batch_2.mat');
[X3,Y3,y3] = LoadBatch('data_batch_3.mat');
[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat');

%% Multi SVM
clear all
load MNISTData.mat

datasize = 300;
testSample = 30;

kernelType = 'rad'; % 'pol' // 'lin' // 'rad'. Just nu jobbar vi med kvadratisk Kernel
kernelParameter = 200; % Takes contextual value depending on Kernel choosen. But can be either p or sigma

% Hämta data
features = imagesTrain(:,1:datasize);
labels = labelsTrain(1:datasize,1)' + 1; % +1 kompenserar för att värderna är noll skjusterade.

%Klassificera
res =  multiSVM(features, labels, kernelType, Inf, 10e-9, kernelParameter);
testL = classifySVM(imagesTest(:,1:testSample), res, kernelType, kernelParameter);

% Beräkna fel
error = 1 - nnz(testL == labelsTest(1:testSample)'+1)/size(testL,2)


