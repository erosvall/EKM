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

% --------------------------
datasize = [5000:5000:5000];
kernelparam = 1;
lambda= 10;
% --------------------------
for d = datasize
    
    X = imagesTrain(:,1:d);
    L = labelsTrain(1:d,1)'+1;
    hiddenNodes = size(X,1)*2;

   
    % Train model
    t = cputime();
    [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
    trainTime = [trainTime, cputime - t];
    
    % Test model
    t = cputime();
    testL = KELMclassifier(imagesTest,sigma,wi,wo,kernel,kernelparam); 
    classificationTime  = [classificationTime, cputime - t];
    
    accuracy = [accuracy, nnz(testL == labelsTest'+1)/size(testL,2)];

           
                
end
acc = accuracy;
classTime = classificationTime;

clear  accuracy d classificationTime tempTrainTime imagesTrain imagesTest labelsTest labelsTrain sigma testL wi wo X L t
save(strcat('MNIST',kernel,num2str(kernelparam)));
clear all
toc
%% RANDOM FACES AR
kernel = 'rbf';
datasize = [0.3:0.05:0.8];
kernelparam = [1];
lambda = [1e0];
iterations = 2;
%--------------------------

rng(420)
tic
disp('AR')
load randomfaces4AR.mat

acc = [];
trainTime = [];
classTime = [];

N = size(featureMat,2);

for j = 1:iterations
    % suffle data
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);   
    
    % temp save vectors
    tempAcc = [];
    tempTrainTime = [];
    tempClassTime = [];
    
    for d = datasize

        X = featureMat(:,1:round(N*d));
        L = labelMat(:,1:round(N*d));

        hiddenNodes = size(X,1)*2;
        
        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);                
        tempTrainTime = [tempTrainTime, cputime - t];


        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*d) + 1:end),sigma,wi,wo,kernel,kernelparam);
        tempClassTime  = [tempClassTime, cputime - t];

        [~,class] = max(labelMat(:,round(N*d) + 1:end));

        tempAcc = [tempAcc, nnz(testL == class)/size(testL,2)];    
                
    end
    acc = [acc; tempAcc];
    trainTime = [trainTime; tempTrainTime];
    classTime = [classTime; tempClassTime];
end

acc = mean(acc)
trainTime = mean(trainTime);
classTime = mean(classTime);
clear tempAcc tempClassTime tempTrainTime featureMat filenameMat labelMat a timeTrain timeTrain wi wo sigma X L class d iteration j N t testL
save(strcat('AR',kernel,num2str(kernelparam)))
clear all
toc
%% YALEFACES EXTENDED
kernel = 'rbf';
kernelparam = 1;
lambda = 1e8;
datasize = 0.3:0.1:0.8;
iterations = 2;

tic
disp('yalefaces')
load randomfaces4extendedyaleb.mat

N = size(featureMat, 2);

rng(420)
acc = [];
trainTime = [];
classTime = [];
for j = 1:iterations
    % Rearrange all the classes
    a = randperm(N);
    featureMat = featureMat(:,a);
    labelMat = labelMat(:,a);    
    
    % create temp storage
    tempAcc = [];
    tempTrainTime = [];
    tempClassTime = [];
    for i = datasize
        X = featureMat(:,1:round(N*i));
        L = labelMat(:,1:round(N*i));

        hiddenNodes = size(X,1)*2;
        lambda = 1e8;

        t = cputime();
        [wi, wo, sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);
        tempTrainTime = [tempTrainTime, cputime - t];

        %l = KELMclassifier(X,sigma,wi,wo,kernel,1);
        %trainAccuracy = nnz(l == L)/size(l,2)
        t = cputime();
        testL = KELMclassifier(featureMat(:,round(N*i) + 1:end),sigma,wi,wo,kernel,kernelparam);
        tempClassTime  = [tempClassTime, cputime - t];

        [~,class] = max(labelMat(:,round(N*i) + 1:end));

        tempAcc = [tempAcc, nnz(testL == class)/size(testL,2)];    
    end
    acc = [acc;tempAcc];
    trainTime = [trainTime;tempTrainTime];
    classTime = [classTime;tempClassTime];
end

acc = mean(acc);
trainTime = mean(trainTime);
classTime = mean(classTime);

clear featureMat filenameMat labelMat class i j N t  tempAcc tempClassTime tempTrainTime testL a  wi wo sigma X L
save(strcat('YF',kernel,num2str(kernelparam)))
clear all
toc
%% CALTECH101
kernel = 'rbf';
kernelparam = 1;
lambda = 1e8;
datasize = 0.3:0.1:0.8;
iterations = 2;
% ----------------

tic
disp('caltech')
load spatialpyramidfeatures4caltech101.mat
N = size(featureMat, 2);
rng(420)

CalTech101resAcc = [];
CalTech101resTrainTime = [];
CalTech101resClassTime = [];


for j = 1:iterations
    % Rearrange all the classes
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
acc = mean(CalTech101resAcc);
classTime = mean(CalTech101resClassTime);
trainTime = mean(CalTech101resTrainTime);

clear featureMat filenameMat labelMat CalTech101resAcc timeTrain timeClass CalTech101resClassTime class i j N t testL CalTech101resTrainTime a wi wo sigma X L
save(strcat('CalTech101',kernel,num2str(kernelparam)))
clear all
toc
%% SCENE15
kernel = 'rbf';
kernelparam = 1;
lambda = 1e-7;
datasize = 0.5:0.1:0.6;
iterations = 2;
% ----------------

tic
load spatialpyramidfeatures4scene15.mat
disp('scene15')
N = size(featureMat, 2);
rng(420)
scene15resAcc = [];
scene15resTrainTime = [];
scene15resClassTime = [];


for j = 1:iterations
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
acc = mean(scene15resAcc);
trainTime = mean(scene15resTrainTime);
classTime = mean(scene15resClassTime);

clear featureMat filenameMat scene15resClassTime scene15resTrainTime timeClass scene15resAcc class i j N t testL labelMat a timeTrain timeTrain wi wo sigma X L
save(strcat('scene15',kernel,num2str(kernelparam)))
clear all
toc