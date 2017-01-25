%% Add Images to working directory
% Make sure path is correct for your computer. But not you Viktor
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/training/';
%path = '/Users/Viktor/Dropbox/KTH/�r 3/Period 3/Kex/yalefaces/training/';
addpath(genpath(path));




%% Generate training system
clc

%mshow('s1p1.gif');
imageVector = []; 
%imageVector = vpa(imageVector);
MU = [];
SIGMA = [];
numbOfClasses = 9;
downSampling = 777;
c = 10e-3;

A = rand(512,77760);
fileEnding = {'centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy'};
for i = 1:numbOfClasses
    mu = [];
    for k = 1:9
        imgFileName = strcat('subject0',num2str(i),'.',fileEnding{1,k}); 
        img = imread(imgFileName);
        j = im2double(img);
        imageVector = [imageVector resample(j(:),1,downSampling)];
    end
    
    for l = 1:length(imageVector(:,1))
         mu = [mu ; mean(imageVector(l,:))];
    end
    
    SIGMA(:,:,i) = cov(imageVector') + c*eye(max(size(imageVector))); 
    %IV primmas f�r att:
    %For matrices, where each row is an observation, 
    %and each column a variable, cov(X) is the 
    %covariance matrix
    
    MU = [MU mu];
    
    disp('Done with class')
end
disp('Done with modelling')
%% Generate test system
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/test/';
%path = '/Users/Viktor/Dropbox/KTH/�r 3/Period 3/Kex/yalefaces/test/';
prob = [];

addpath(genpath(path));

fileEndingTest = {'wink','surprised'};
fileDataTest = string(fileEndingTest);

imgFileName = strcat('subject02.wink');
img = imread(imgFileName);

testImage = im2double(img);
testVector = resample(testImage(:),1,downSampling);

prob = [];

for i = 1:numbOfClasses
    prob = [prob mvnpdf(testVector, MU(:,i), SIGMA(:,:,i))];
    cond(SIGMA(:,:,i))
end

disp(prob);