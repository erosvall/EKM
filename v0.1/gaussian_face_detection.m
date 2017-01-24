%% Add Images to working directory
% Make sure path is correct for your computer. But not you Viktor
path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/training/';
addpath(genpath(path));


%% Generate training system
imshow('subject01.gif');
img = imread('subject01.gif');
imgvector = img(:);
%&GMM = fitgmdist(imgvector,500);

% Read files file1.txt through file20.txt, mat1.mat through mat20.mat
% and image1.jpg through image20.jpg.  Files are in the current directory.
for k = 1:20
  matFilename = sprintf('mat%d.mat', k);
  matData = load(matFilename);
  jpgFilename = strcat('image', num2str(k), '.jpg');
  imageData = imread(jpgFilename);
  textFilename = ['file' num2str(k) '.txt'];
  fid = fopen(textFilename, 'rt');
  textData = fread(fid);
  fclose(fid);
end


for i = 1:15
    for k = 1:9
        
    end
end
%% Generate test system
