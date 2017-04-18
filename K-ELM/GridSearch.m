
load('MNISTData.mat')
datasize = 5000;

X = imagesTrain(:,1:datasize);
Xt = imagesTest;

Y = labelsTrain(1:datasize)'+1;
Yt = labelsTest'+1;

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




