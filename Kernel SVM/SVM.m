%% Generate data to seperate
% This function will work with data that has coordinates and class appended
% into one matrix. For theoretical examples we'll not use fitrsvm() method
% that solves this particular problem?
close all
D = NormalDataGeneration(2,30,2);
scatter(D(:,1),D(:,2))
class = [ones(size(D,1)/2,1); ones(size(D,1)/2,1) *-1];
D = [D class];


%% Determine Kernel function
% Radial Kernel "radKerl.m"
% Linear Kernel "linKerl.m"
% Polynomial Kernel "polKerl.m"

%% Classification Function - returns -1 or 1 depending on class
% Takes NonZeroAlphas and Points to look at and classifies them

%% Calculate SVM with given Kernel
n = size(D,1); % Length of data
P = zeros(n,n);
for i = 1:n
    for j = 1:n
        d1 = D(i,:);
        d2 = D(j,:);
        P(i,j) = d1(3)*d2(3)*linKerl(d1(1,1:end-1),d2(1,1:end-1));
    end
end
q = -ones(n,1);
h = zeros(2*n,1); % Constraint on optimization criterion Ax =< b
G = -eye(2*n,n);
r = quadprog(P,q,G,h);


%%
threshold = 10^(-05);
slackPressure = 10^(2);
nonZeroAlpha = find(r > threshold & r < slackPressure);
res = [];
k = 1;
for i = nonZeroAlpha'
    res(k,:) = [r(i), D(i,:)];
    k = k + 1;
end

%% Verify with datapoint
testPoint = [19,10];
sign(classifySVM(testPoint,res,'lin'))