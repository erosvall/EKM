%% K-SVD Classifier
clear all
close all

trainsize = 400;
valsize = 100;

A = load('MNISTData.mat');
Y = A.imagesTrain(:,1:trainsize); y = A.labelsTrain(1:trainsize)+1;
Yv = A.imagesTest(:,1:valsize); yv = A.labelsTest(1:valsize)+1;

DictionarySize = 1960;
UpdateIterations = 20;
Lambda = 0.4;
Sparcity = 5;

[D,W] = KSVD_Classifier(Y,y,DictionarySize,UpdateIterations,Lambda,Sparcity);

[trainLabels,sparsedata] = KSVD_Labeler(Y,D,W,Sparcity);
TrainAccuracy = nnz(trainLabels == y')/size(y,1)

testLabels = KSVD_Labeler(Yv,D,W,Sparcity);
TestAccuracy = nnz(testLabels == yv')/size(yv,1)

%% Visualize 
figure
realPics = Y;
%realPics = reshape(realPics,[28,28,200]);
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(realPics(:,i),28)');
end
figure
X = OMP(D,Y,5);
sparcePics = D*X;
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(sparcePics(:,i),28)');
end

figure
testPics = Yv;
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(testPics(:,i),28)');
end

Xv = OMP(D,Yv,5);
figure
sparceTestPics = D*Xv;
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(sparceTestPics(:,i),28)');
end




