%% Least Square classifier
close all 
clear all

D =  NormalDataGeneration(2,10,2);
scatter(D(:,1),D(:,2))

%% Learning

close all

%Labeling data
b = [ones(size(D,1)/2,1) ; -1*ones(size(D,1)/2,1)];
A = [D ones(size(D,1),1)];

%hyperplane coefficients
a =  pinv(A)*b ;

x = [0,20];
y = (-a(3) - a(1)*x)./a(2);
plot(x,y)

hold on
scatter(D(:,1),D(:,2))
hold on

%% Classification 
normal = a(1:2)/norm(a(1:2),2);
d = 5; % number of new points

newData = [NormalDataGeneration(1,d,2) zeros(d,1)];

newData(:,3) = sign(newData(1,1:2)*normal);
scatter(newData(:,1),newData(:,2),'b');
newData


