%% Add Images to working directory
% Make sure path is correct for your computer. But not you Viktor
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/mattraining/';
addpath(genpath(path));


%% Generate training system
clc
%mshow('s1p1.gif');
IV = []; 
A = rand(100,77760);
for i = 1:1
    for k = 1:10
        imgFileName = strcat('s',num2str(i),'p',num2str(k),'.gif');
        img = imread(imgFileName);
        j = im2double(img);
        IV = [IV A*j(:)];
    end
    
    options = statset('Display','final');
    GMM = fitgmdist(IV,1,'Options',options,'Regularize',0.00001);

end

%% Generate test system
