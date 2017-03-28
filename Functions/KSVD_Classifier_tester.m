%% K-SVD Classifier
%clear all
%close all

trainsize = 300;
valsize = 100;

A = load('MNISTData.mat');
Y = A.imagesTrain(:,1:trainsize);
y = A.labelsTrain(1:trainsize)+1;
Yv = A.imagesTest(:,1:valsize);
yv = A.labelsTest(1:valsize)+1;


[D1,W1] = KSVD_Classifier(Y,y,1960,20,0.4,5);


%% Compute accuracy.
X = OMP(D,Y,5);
T = W*X;
[~,trainmaxarg] = max(T);
Trainacc = nnz(trainmaxarg == y')/size(y,1)

Xv = OMP(D,Yv,5); % Finding the sparce rep
That = W*Xv;
[~,testmaxarg] = max(That);
Testacc = nnz(testmaxarg == yv')/size(yv,1)

%% Visualize 
figure
realPics = Y;
%realPics = reshape(realPics,[28,28,200]);
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(realPics(:,i),28)');
end

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




