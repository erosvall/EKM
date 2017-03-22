    
path = '/Users/Viktor/Dropbox/KTH/År 3/Period 3/Kex/yalefaces/training/';
addpath(genpath(path));

imageVector = []; 
labVector = [];

numbOfClasses = 15;
    
fileEnding = {'sad','glasses','happy','leftlight','noglasses','centerlight','rightlight','normal','sleepy'};
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
        imageVector = [imageVector j(:)];
        labVector = [labVector i];
    end
end
 
%%
path = '/Users/Viktor/Dropbox/KTH/År 3/Period 3/Kex/yalefaces/test/';
addpath(genpath(path));
fileEndingTest = {'wink','surprised'};
fileDataTest = string(fileEndingTest);

testLabVector = [];
testVector = [];
for i = 1:2 %looping over fileEndings

    for j = 1:numbOfClasses % for every person 
        %disp(strcat('person: ',num2str(j)))
        

        if(j<10)
            imgFileName = strcat('subject0',num2str(j),'.',fileEndingTest{1,i}); 
        else
            imgFileName = strcat('subject',num2str(j),'.',fileEndingTest{1,i}); 
        end
        img = imread(imgFileName);
        testImage = im2double(img);
        testVector = [testVector testImage(:)];
        testLabVector = [testLabVector j];

    end
end
%%
features = [imageVector, testVector];
labels = [labVector, testLabVector];