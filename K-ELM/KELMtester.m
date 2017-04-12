
clear all
load MNISTData.mat

datasize = 10000;  

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)'+1;

hiddenNodes = size(X,1)*2;
lambda = 1;

[wi, wo] = KELMtrainer(X,L,hiddenNodes,lambda,'rbf',1);

l = KELMclassifier(X,X,wi,wo,'rbf',1);
trainAccuracy = nnz(l == L)/size(l,2)

testL = KELMclassifier(imagesTest,X,wi,wo,'rbf',1);
testAccuracy = nnz(testL == labelsTest'+1)/size(testL,2)