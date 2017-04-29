%% Code for testing the effect of the number of hidden nodes

%% RANDOM FACES AR
kernel = 'poly';
kernelparam = 1;
datasize = 0.3:0.05:0.8;
epsilon = 1e9;
iterations = 100;
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