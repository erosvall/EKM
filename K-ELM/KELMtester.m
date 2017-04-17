
clear all
load MNISTData.mat

datasize = 5000;  

X = imagesTrain(:,1:datasize);
L = labelsTrain(1:datasize,1)'+1;

hiddenNodes = size(X,1)*2;
lambda = 10;

kernel = 'poly';
kernelparam = 2;

[wi, wo,sigma] = KELMtrainer(X,L,hiddenNodes,lambda,kernel,kernelparam);

l = KELMclassifier(X,sigma,wi,wo,kernel,kernelparam);
trainAccuracy = nnz(l == L)/size(l,2)

testL = KELMclassifier(imagesTest,sigma,wi,wo,kernel,kernelparam);
testAccuracy = nnz(testL == labelsTest'+1)/size(testL,2)