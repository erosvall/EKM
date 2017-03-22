%% MNIST Data generation 

imagesTrain = loadMNISTImages('train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels.idx1-ubyte');
trainData = [labelsTrain imagesTrain'];
subTrainData = trainData(1:100,1:end);


imagesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');
testData = [labelsTest imagesTest'];
%dlmwrite('testing.txt',testData);

for i = 1:10
    subplot(4,5,i)
    imshow(vec2mat(imagesTrain(:,i),28)');
end
labels(1:10,1);

