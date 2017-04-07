
%% ELM on MNIST 60 000 training, 10 000 testing
clear all
load MNISTData.mat

datasize = 10000;  

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

%% KSVD-Classifier MNIST
close all
valsize = 10000;
trainsize = 60000;
DictionarySize = 1500;

UpdateIterations = 10;
Lambda = 0.4;
Sparcity = 5;
Acc = [];

A = load('MNISTData.mat');
for Sparcity = 5
    Y = A.imagesTrain(:,1:trainsize); y = A.labelsTrain(1:trainsize)+1;
    Yv = A.imagesTest(:,1:valsize); yv = A.labelsTest(1:valsize)+1;
    
    [D,W] = KSVD_Classifier(Y,y,DictionarySize,UpdateIterations,Lambda,Sparcity);

    trainLabels = KSVD_Labeler(Y,D,W,Sparcity);
    TrainAccuracy = nnz(trainLabels == y')/size(y,1)

    testLabels = KSVD_Labeler(Yv,D,W,Sparcity);
    TestAccuracy = nnz(testLabels == yv')/size(yv,1)
    Acc = [Acc; Sparcity, TrainAccuracy, TestAccuracy];
    Sparcity
end
%%
scatter(Acc(:,1),Acc(:,2))
hold on
scatter(Acc(:,1),Acc(:,3))
legend('Training data', 'Validation data')
xlabel('Lambda')
ylabel('Accuracy')

%% KSVD-Classifier CIFAR-10
clear all
close all

trainsize = 500;
valsize = 100;

[X1,Y1,y1] = LoadBatch('data_batch_1.mat');
[X2,Y2,y2] = LoadBatch('data_batch_2.mat');
[X3,Y3,y3] = LoadBatch('data_batch_3.mat');
[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat');

Y = X1(:,1:trainsize); y = y1(1:trainsize);
Yv = X2(:,1:valsize); yv = y2(1:valsize);


DictionarySize = 3000;
UpdateIterations = 20;
Lambda = 0.4;
Sparcity = 10;

[D,W] = KSVD_Classifier(Y,y,DictionarySize,UpdateIterations,Lambda,Sparcity);

[trainLabels,sparsedata] = KSVD_Labeler(Y,D,W,Sparcity);
TrainAccuracy = nnz(trainLabels == y')/size(y,1)

testLabels = KSVD_Labeler(Yv,D,W,Sparcity);
TestAccuracy = nnz(testLabels == yv')/size(yv,1)

%% ELM with kernel


clear all
load MNISTData.mat

datasize = 10000;  

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)';

hiddenLayers = 2*size(X,1);
for lambda = 20:1:30
    lambda
    [Wi, Wo] = ELMwithKernelTraining(X,L,hiddenLayers,lambda);

    kernelTrainL =  ELMwithKernelClassifier(X,Wi,Wo,'rlu');
    kernelTrainAccuracy = nnz(kernelTrainL == L)./size(L,2)

    kernelTestL = ELMwithKernelClassifier(imagesTest,Wi,Wo,'rlu');
    kernelTestAccuracy = (nnz(kernelTestL == labelsTest')/size(kernelTestL,2))

    disp('-----------')
end
% [W1,W0] = ELMtrain(X,L,hiddenLayers);
% 
% 
% trainL = ELMclassifier(X,W1,W0,'rlu');
% trainAccuracy = (nnz(trainL == L)/size(L,2))
% 
% testL = ELMclassifier(imagesTest,W1,W0,'rlu');
% testAccuracy = (nnz(testL == labelsTest')/size(testL,2))


%% Visualize CIFAR
close all

for i=1:10
    im = reshape(Y(:,i), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(1)
montage(s_im);

X = OMP(D,Y,5);
sparcePics = D*X;
for i=1:10
    im = reshape(sparcePics(:,i), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(2)
montage(s_im);

testPics = Yv;
for i=1:10
    im = reshape(testPics(:,i), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(3)
montage(s_im);

Xv = OMP(D,Yv,5);
sparceTestPics = D*Xv;
for i=1:10
    im = reshape(sparceTestPics(:,i), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(4)
montage(s_im);

