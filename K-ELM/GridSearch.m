clear all
load('MNISTData.mat')
datasize = 5000;

X = imagesTrain(:,1:datasize);
Xt = imagesTest;

Y = labelsTrain(1:datasize)'+1;
Yt = labelsTest'+1;


%% För alla andra dataset
clear all
%load('randomfaces4AR')
%load('randomfaces4extendedyaleb')
load('spatialpyramidfeatures4scene15')
%load('spatialpyramidfeatures4caltech101')

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


saveParams = [];
%% KELM Grid serach
kernel = 'rbf';
lambda = 10.^[-9:1:-5];
kernelparam = 10.^[-2:1:2];
h = size(X,1)*2;

Accuracy = [];
params = [];


for l = lambda
    for kp = kernelparam        
        tic
        %Computing accuracy
        Accuracy = [Accuracy; l kp KELMClassificationAccuracy(X,Y,Xt,Yt,l,h,kernel,kp)];
        params = [params;l kp];
        toc
    end
end
[maxAccuracy,maxIndex] = max(Accuracy(:,3));
maxAccuracy
topParams = params(maxIndex,:);

saveParams = [saveParams; maxAccuracy topParams];


