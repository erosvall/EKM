clear all

kernel = 'poly';
kernelparam = 1;

%cd /Users/erikrosvall/github/KEX/K-ELM/
addpath('C:\Users\Viktor Karlsson\Dropbox\KTH\År 3\Period 4\Kex\Datasets');
%% MNIST
tic
disp('mnist')
load MNISTData.mat
MNISTacc = [];
MNISTtimeTrain = [];
MNISTtimeClass = [];
datasize = 5000:5000:30000;
for i = datasize
    X = imagesTrain(:,1:i);
    L = labelsTrain(1:i,1)'+1;

    hiddenNodes = size(X,1)*2;
    lambda = 1e4;
    
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
save(strcat('MNIST',kernel,'_p=1'));
toc
%% RANDOM FACES AR
tic
disp('AR')
load randomfaces4AR.mat

datasize = 0.3:0.05:0.8;
% Rearrange all the classes
rng(420)
N = size(featureMat, 2);
ARresAcc = [];
ARresTrainTime = [];
ARresClassTime = [];
for j = 1:100
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
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
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
save(strcat('AR',kernel,'_p=1'))
toc
%% YALEFACES EXTENDED
tic
disp('yalefaces')
load randomfaces4extendedyaleb.mat


N = size(featureMat, 2);
datasize = 0.3:0.05:0.8;
% Rearrange all the classes
rng(420)


YFresAcc = [];
YFresTrainTime = [];
YFresClassTime = [];


for j = 1:100
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
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
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
save(strcat('YF',kernel,'_p=1'))

toc
%% CALTECH101
tic
disp('caltech')
load spatialpyramidfeatures4caltech101.mat


N = size(featureMat, 2);
datasize = 0.3:0.05:0.8;
% Rearrange all the classes
rng(420)


CalTech101resAcc = [];
CalTech101resTrainTime = [];
CalTech101resClassTime = [];


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
        lambda = 1e-2;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
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
save(strcat('CalTech101',kernel,'_p=1'))
toc
%% SCENE15
tic
load spatialpyramidfeatures4scene15.mat
disp('scene15')

N = size(featureMat, 2);
datasize = 0.3:0.05:0.8;
% Rearrange all the classes
rng(420)


scene15resAcc = [];
scene15resTrainTime = [];
scene15resClassTime = [];


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
        lambda = 1e-8;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
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
save(strcat('scene15',kernel,'_p=1'))
toc