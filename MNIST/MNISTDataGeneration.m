%% MNIST Data generation 

imagesTrain = loadMNISTImages('train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels.idx1-ubyte');
trainData = [labelsTrain imagesTrain'];
trainData(1:2,1:10)

imagesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');
testData = [labelsTest imagesTest'];

for i = 1:10
    subplot(4,5,i)
    imshow(vec2mat(images(:,i),28)');
end
labels(1:10,1);

