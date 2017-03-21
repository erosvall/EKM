imagesTrain = loadMNISTImages('train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels.idx1-ubyte');

imagesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');
%% ELM
datasize = 6000;

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)';
[Wi, Wo] = ELMtrain(X,L,1000);

testL = ELMclassifier(imagesTest,Wi,Wo);
accuracy = nnz(testL == labelsTest')/size(testL,2)

%% 
