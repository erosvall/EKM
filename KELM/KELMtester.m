
clear all
load MNISTData.mat

datasize = 20000;  

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)'+1;

hiddenNodes = size(X,1)*2;
lambda = 1;


[wi, wo] = KELM(X,L,hiddenNodes,lambda);
size(wi)
size(wo)
size(X)
l = KELMclassifier(X,X,wi,wo);
trainAccuracy = nnz(l == L)/size(l,2)

testL = KELMclassifier(imagesTest,X,wi,wo);
testAccuracy = nnz(testL == labelsTest'+1)/size(testL,2)