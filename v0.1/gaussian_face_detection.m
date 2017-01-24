%% Add Images to working directory
% Make sure path is correct for your computer. But not you Viktor
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/training/';
addpath(genpath(path));


%% Generate training system
clc
%mshow('s1p1.gif');
IV = []; 
A = rand(8,77760);
fileEnding = {'centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy'};
for i = 1:1
    for k = 1:9
        imgFileName = strcat('subject0',num2str(i),'.',fileEnding{1,k}); 
        img = imread(imgFileName);
        j = im2double(img);
        IV = [IV A*j(:)];
    end
    GMM = fitgmdist(IV',1,'Regularize',0.00001);

end

%% Generate test system
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/test/';
addpath(genpath(path));

fileEndingTest = {'wink','surprised'};
fileDataTest = string(fileEndingTest);

imgFileName = strcat('subject01.wink');
img = imread(imgFileName);
j = im2double(img);
X = A*j(:);

y = mvnpdf(X,GMM.mu, GMM.Sigma);
posterior(GMM,X')