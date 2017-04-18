
load('MNISTData.mat')
datasize = 5000;

X = imagesTrain(:,1:datasize);
Xt = imagesTest;

Y = labelsTrain(1:datasize)'+1;
Yt = labelsTest'+1;


%% För alla andra dataset
addpath('~/Dropbox/Kex/Datasets/Data från Ayman');
load('randomfaces4AR')
% randomfaces4extendedyaleb
% randomfaces4AR
% spatialpyramidfeatures4scene15
% spatialpyramidfeatures4caltech101

N = size(featureMat,2);
% Väljer alltid 70% av datasetet.
% Shuffle för att se till att dataseten blir användvara
a = randperm(N);
featureMat = featureMat(:,a);
labelMat = labelMat(:,a);   

X = featureMat(:,1:round(N*0.7));
Xt = featureMat(:,round(N*0.7) + 1:end);

Y = labelMat(:,1:round(N*0.7));
Yt = labelMat(:,round(N*0.7) + 1:end);



%% KELM Grid serach
kernel = 'rbf';
lambda = [1:2];
kernelparam = [1:10];
Accuracy = [];
for l = lambda
    for kp = kernelparam
        Accuracy = [Accuracy; l kp KELMClassificationAccuracy(X,Y,Xt,Yt,l,kernel,kp)]
    end
end
[maxAccuracy,maxIndex] = max(Accuracy(:,3));




