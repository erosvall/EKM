%% Add Images to working directory
% Make sure path is correct for your computer. But not you Viktor
%path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/training/';
path = '/Users/Viktor/Dropbox/KTH/År 3/Period 3/Kex/yalefaces/training/';
addpath(genpath(path));




%% Generate training system
clc
imageVector = []; 

MU = [];
SIGMA = [];
numbOfClasses = 15;
SizeOfVector = 500; % to high dimension leads to ill-contitiond sigma

c = 10e-3; %solving ill-condition of SIGMA


fileEnding = {'centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy'};
for i = 1:numbOfClasses
    mu = [];
    for k = 1:9
        if(i<10)
            imgFileName = strcat('subject0',num2str(i),'.',fileEnding{1,k}); 
        else
            imgFileName = strcat('subject',num2str(i),'.',fileEnding{1,k}); 
        end
        img = imread(imgFileName);
        j = im2double(img);
        imageVector = [imageVector resample(j(:),SizeOfVector,max(size(j(:))))];
    end
    
    for l = 1:length(imageVector(:,1))
         mu = [mu ; mean(imageVector(l,:))];
    end
    
    SIGMA(:,:,i) = cov(imageVector') + c*eye(max(size(imageVector))); 
    %IV primmas för att:
    %For matrices, where each row is an observation, 
    %and each column a variable, cov(X) is the 
    %covariance matrix
    
    MU = [MU mu];
    
    disp(strcat('Done with class: ',num2str(i)))
end


disp(' ')
disp('Done with modelling!')
disp(' ')

%% Generate test system
%clc
%path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/test/';
path1 = '/Users/Viktor/Dropbox/KTH/År 3/Period 3/Kex/yalefaces/test/';
prob = [];

addpath(genpath(path1));

fileEndingTest = {'wink','surprised'};
fileDataTest = string(fileEndingTest);

for i = 1:2 %looping over fileEndings
    correctCounter = 0;
    
    disp(fileEndingTest{1,i})
    disp('------------')
    for j = 1:numbOfClasses % for every person 
        disp(strcat('person: ',num2str(j)))
        testVector = [];
        prob = [];
        if(j<10)
            imgFileName = strcat('subject0',num2str(j),'.',fileEndingTest{1,i}); 
        else
            imgFileName = strcat('subject',num2str(j),'.',fileEndingTest{1,i}); 
        end
        %imgFileName = strcat('subject0',num2str(j),'.',fileEndingTest{1,i}); 
        img = imread(imgFileName);
        testImage = im2double(img);
        testVector = resample(testImage(:),SizeOfVector,77760);

        for k = 1:numbOfClasses %looping through the model
            prob = [prob mvnpdf(testVector, MU(:,k), SIGMA(:,:,k))];
            
        end
        
        disp(strcat('classified as person: ', num2str(find(prob == max(prob(:))))))
        
        if(j == find(prob == max(prob(:))))
            correctCounter = correctCounter + 1;
        end
    end
    disp(' ' )
    disp(strcat('Accuracy of classification:', num2str(100*correctCounter/numbOfClasses),'%'))
    disp('------------')
end
%% Test på bilder i träningsdata
%path = '/Users/erikrosvall/Dropbox/Kex/yalefaces/test/';
prob = [];

addpath(genpath(path));

fileEndingTest = {'centerlight','glasses'};
fileDataTest = string(fileEndingTest);

for i = 1:2 %looping over fileEndings
    correctCounter = 0;
    
    disp(fileEndingTest{1,i})
    disp('------------')
    for j = 1:numbOfClasses % for every person 
        disp(strcat('person: ',num2str(j)))
        testVector = [];
        prob = [];
        if(j<10)
            imgFileName = strcat('subject0',num2str(j),'.',fileEndingTest{1,i}); 
        else
            imgFileName = strcat('subject',num2str(j),'.',fileEndingTest{1,i}); 
        end
        %imgFileName = strcat('subject0',num2str(j),'.',fileEndingTest{1,i}); 
        img = imread(imgFileName);
        testImage = im2double(img);
        testVector = resample(testImage(:),SizeOfVector,77760);

        for k = 1:numbOfClasses %looping through the model
            prob = [prob mvnpdf(testVector, MU(:,k), SIGMA(:,:,k))];
            
        end
        
        disp(strcat('classified as person: ', num2str(find(prob == max(prob(:))))))
        
        if(j == find(prob == max(prob(:))))
            correctCounter = correctCounter + 1;
        end
    end
    disp(' ' )
    disp(strcat('Accuracy of classification:', num2str(100*correctCounter/numbOfClasses),'%'))
    disp('------------')
end




