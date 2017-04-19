clear all

kernel = 'rbf';
kernelparam = 1;

%cd /Users/erikrosvall/github/KEX/K-ELM/
%addpath('C:\Users\Viktor Karlsson\Dropbox\KTH\År 3\Period 4\Kex\Datasets');
%% MNIST
kernel = 'rbf';

tic
disp('mnist')
load MNISTData.mat

accuracy = [];
trainTime = [];
classificationTime = [];
gridSearchSave=[];
% --------------------------
datasize = [5000:5000:5000];
kernelparam = [1:2];
lambda = [1e0,1e1];
% --------------------------
for d = datasize
    
    X = imagesTrain(:,1:d);
    L = labelsTrain(1:d,1)'+1;
    hiddenNodes = size(X,1)*2;
    
    for l = lambda    
        for kp = kernelparam
            % Train model
            t = cputime();
            [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,l,kernel,kp);
            tempTrainTime = cputime - t;
            trainTime = [trainTime, tempTrainTime];

            
            % Test model
            t = cputime();
            testL = KELMclassifier(imagesTest,sigma,wi,wo,kernel,kp);
            tempClassificationTime = cputime - t;
            classificationTime  = [classificationTime, tempClassificationTime];
            
            tempAccuracy = nnz(testL == labelsTest'+1)/size(testL,2);
            accuracy = [accuracy, tempAccuracy];
            
            gridSearchSave = [gridSearchSave; d l kp tempAccuracy tempTrainTime tempClassificationTime];
        end        
    end
end

[maxAccuracy,maxIndex] = max(gridSearchSave(:,4));

clear imagesTrain imagesTest labelsTest labelsTrain sigma testL wi wo X L t
save(strcat('MNIST',kernel,num2str(kernelparam)));

%clear all
toc
%% RANDOM FACES AR
kernel = 'rbf';
rng(420)

tic
disp('AR')
load randomfaces4AR.mat

accuracy = [];
trainTime = [];
classificationTime = [];
gridSearchSave=[];
%--------------------------
datasize = [0.3:0.05:0.8];
kernelparam = [1:2];
lambda = [1e0,1e1];
iterations = 100;
%--------------------------
for j = 1:iterations
    a = randperm(size(featureMat, 2));
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);   
    
    tempAcc = [];
    tempTrainTime = [];
    tempClassificationTime = [];
    for d = datasize
        for l = lambda
            for kp = kernelparam
                X = featureMat(:,1:round(N*d));
                L = labelMat(:,1:round(N*d));

                hiddenNodes = size(X,1)*2;
                lambda = 1e21;

                t = cputime();
                [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,l,kernel,kp);
                tempTime
                timeTrain = [timeTrain, cputime - t];

                %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
                %trainAccuracy = nnz(l == L)/size(l,2)
                t = cputime();
                testL = KELMclassifier(featureMat(:,round(N*d) + 1:end),sigma,wi,wo,kernel,kp);
                timeClass  = [timeClass, cputime - t];

                [~,class] = max(labelMat(:,round(N*d) + 1:end));

                acc = [acc, nnz(testL == class)/size(testL,2)];    
            end
        end                
    end
    ARresAcc = [ARresAcc; acc];
    ARresTrainTime = [ARresTrainTime; timeTrain];
    ARresClassTime = [ARresClassTime; timeClass];
end

clear featureMat filenameMat labelMat a timeTrain acc timeTrain wi wo sigma X L
save(strcat('AR',kernel,'_p=1'))
clear ARresAcc ARresTrainTime ARresClassTime timeClass testL t N lambda j i hiddenNodes datasize class
toc
%% YALEFACES EXTENDED
tic
disp('yalefaces')
load randomfaces4extendedyaleb.mat


N = size(featureMat, 2);
datasize = 0.3:0.1:0.8;
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
        lambda = 1e8;

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
clear YFresAcc YFresTrainTime YFresClassTime class datasize hiddenNodes i j lambda N t testL timeClass
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
        lambda = 1e-1;

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
clear CalTech101resAcc CalTech101resClassTime CalTech101resTrainTime class datasize hiddenNodes i j lambda N t testL timeClass

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
        lambda = 1e-7;

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
save(strcat('scene15',kernel))
clear class datasize hiddenNodes i j lambda N scene15resAcc scene15resClassTime scene15resTrainTime t testL timeClass
toc