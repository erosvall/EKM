clear all

cd /Users/erikrosvall/github/KEX/ELM/
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
    [wi, wo] = ELMtrain(X,L,hiddenNodes,lambda);
    MNISTtimeTrain = [MNISTtimeTrain, cputime - t];
    
    %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
    %trainAccuracy = nnz(l == L)/size(l,2)
    t = cputime();
    testL = ELMclassifier(imagesTest,wi,wo);
    MNISTtimeClass  = [MNISTtimeClass, cputime - t];
    
    
    MNISTacc = [MNISTacc, nnz(testL == labelsTest'+1)/size(testL,2)];    
end

clear imagesTrain imagesTest labelsTest labelsTrain sigma testL wi wo X L t
save('MNIST_ELM_with_penalty');

%% RANDOM FACES AR

load randomfaces4AR.mat

datasize = 0.1:0.1:0.2;
% Rearrange all the classes
rng(420)
N = size(featureMat, 2);
ARresAcc = [];
ARresTrainTime = [];
ARresClassTime = [];
for j = 1:2
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);   
    
    acc = [];
    timeTrain = [];
    timeClass = [];
    for i = datasize
        X = featureMat(:,1:round(N*i));
        L = labelMat(:,1:round(N*i));

        hiddenNodes = size(X,1)*2;
        lambda = 1e0;

        t = cputime();
        [wi, wo] = ELMtrain(X,L,hiddenNodes,lambda);
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = ELMclassifier(featureMat(:,round(N*i) + 1:end),wi,wo);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    ARresAcc = [ARresAcc; acc];
    ARresTrainTime = [ARresTrainTime; timeTrain];
    ARresClassTime = [ARresClassTime; timeClass];
end

clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save('AR_ELM_with_penalty')

%% YALEFACES EXTENDED

load randomfaces4extendedyaleb.mat


N = size(featureMat, 2);
datasize = 0.1:0.1:0.9;
% Rearrange all the classes
rng(420)


YFresAcc = [];
YFresTrainTime = [];
YFresClassTime = [];


for j = 1:1
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);    
    acc = [];
    timeTrain = [];
    timeClass = [];
    for i = datasize
        X = featureMat(:,1:round(N*i));
        L = labelMat(:,1:round(N*i));

        hiddenNodes = size(X,1)*2;
        lambda = 1e1;

        t = cputime();
        [wi, wo] = ELMtrain(X,L,hiddenNodes,lambda);
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = ELMclassifier(featureMat(:,round(N*i) + 1:end),wi,wo);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    YFresAcc = [YFresAcc;acc];
    YFresTrainTime = [YFresTrainTime;timeTrain];
    YFresClassTime = [YFresClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save('YF_ELM_with_penalty')


%% CALTECH101
load spatialpyramidfeatures4caltech101.mat


N = size(featureMat, 2);
datasize = 0.7:0.1:0.8;
% Rearrange all the classes
rng(420)


CalTech101resAcc = [];
CalTech101resTrainTime = [];
CalTech101resClassTime = [];


for j = 1:2
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);    
    acc = [];
    timeTrain = [];
    timeClass = [];
    for i = datasize
        X = featureMat(:,1:round(N*i));
        L = labelMat(:,1:round(N*i));

        hiddenNodes = size(X,1)*2;
        lambda = 0.01;

        t = cputime();
        [wi, wo] = ELMtrain(X,L,hiddenNodes,lambda);
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = ELMclassifier(featureMat(:,round(N*i) + 1:end),wi,wo);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    CalTech101resAcc = [CalTech101resAcc;acc];
    CalTech101resTrainTime = [CalTech101resTrainTime;timeTrain];
    CalTech101resClassTime = [CalTech101resClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save('CalTech101_ELM_with_penalty')

%% SCENE15
load spatialpyramidfeatures4scene15.mat


N = size(featureMat, 2);
datasize = 0.1:0.1:0.1;
% Rearrange all the classes
rng(420)


scene15resAcc = [];
scene15resTrainTime = [];
scene15resClassTime = [];


for j = 1:1
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);    
    acc = [];
    timeTrain = [];
    timeClass = [];
    for i = datasize
        X = featureMat(:,1:round(N*i));
        L = labelMat(:,1:round(N*i));

        hiddenNodes = size(X,1)*2;
        lambda = 1e-8;

        t = cputime();
        [wi, wo] = ELMtrain(X,L,hiddenNodes,lambda);
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = ELMclassifier(featureMat(:,round(N*i) + 1:end),wi,wo);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    scene15resAcc = [scene15resAcc;acc];
    scene15resTrainTime = [scene15resTrainTime;timeTrain];
    scene15resClassTime = [scene15resClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save('scene15_ELM_with_penalty')