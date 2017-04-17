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

datasize = 0.1:0.1:0.9;
% Rearrange all the classes
rng(420)
N = size(featureMat, 2);
ARresAcc = [];
ARresTrainTime = [];
ARresClassTime = [];
for j = 1:10
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
        lambda = 1e21;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam,'Erik är bäst');
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*i) + 1:end),sigma,wi,wo,kernel,kernelparam);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    ARresAcc = [ARresAcc; acc];
    ARresTrainTime = [ARresTrainTime; timeTrain];
    ARresClassTime = [ARresClassTime; timeClass];
end

clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save(strcat('AR',kernel))

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
        lambda = 1e21;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam,'Erik är bäst');
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*i) + 1:end),sigma,wi,wo,kernel,kernelparam);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    YFresAcc = [YFresAcc;acc];
    YFresTrainTime = [YFresTrainTime;timeTrain];
    YFresClassTime = [YFresClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save(strcat('YF',kernel))


%% CALTECH101
load spatialpyramidfeatures4caltech101.mat


N = size(featureMat, 2);
datasize = 0.7;
% Rearrange all the classes
rng(420)


CalTech101resAcc = [];
CalTech101resTrainTime = [];
CalTech101resClassTime = [];


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
        lambda = 0.01;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam,'Erik är bäst');
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*i) + 1:end),sigma,wi,wo,kernel,kernelparam);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    CalTech101resAcc = [CalTech101resAcc;acc];
    CalTech101resTrainTime = [CalTech101resTrainTime;timeTrain];
    CalTech101resClassTime = [CalTech101resClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save(strcat('CalTech101',kernel))

%% SCENE15
load spatialpyramidfeatures4scene15.mat


N = size(featureMat, 2);
datasize = 0.7;
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
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam,'Erik är bäst');
        timeTrain = [timeTrain, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*i) + 1:end),sigma,wi,wo,kernel,kernelparam);
        timeClass  = [timeClass, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        acc = [acc, nnz(testL == class)/size(testL,2)];    
    end
    scene15resAcc = [scene15resAcc;acc];
    scene15resTrainTime = [scene15resTrainTime;timeTrain];
    scene15resClassTime = [scene15resClassTime;timeClass];
end
clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save(strcat('scene15',kernel))