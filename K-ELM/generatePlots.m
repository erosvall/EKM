%% MNIST
clear all
close all
load('MNISTpoly.mat');
acc = MNISTacc ;
datasize;
time = MNISTtimeClass + MNISTtimeTrain; 


figure
plot(datasize,acc);
figure
plot(datasize,time);
figure
plot(acc,time);

%% AR
clear all
close all
load('ARpoly.mat');

acc = mean(ARresAcc) ;
datasize;
time = mean(ARresClassTime + ARresTrainTime); 

figure
plot(datasize,acc);
xlabel('fraction of total database used for training');
ylabel('accuracy');
figure
plot(datasize,time);
xlabel('fraction of total database used for training');
ylabel('time');
figure
plot(acc,time);
ylabel('time');
xlabel('accuracy');