function [D,W] = KSVD_Classifier(TrainingData, TrainingLabels, DictionarySize, Iterations, Lambda, Sparcity)
%% K-SVD Classifier
% Each training data is represented as a columnvector in TrainingData with
% corresponding labels are found in TrainingLabels which is a rowvector.
% (Note that labels cannot be zero)
% --- User defined constants ---
% DictionarySize:   Number of atoms in dictionary
% Lambdas:          Importance of correcly classifying 
% Sparcity:         Numer of atoms each data is reprecented 
% Dependencies:
% OMP.m and SVDDictionaryUpdate.m

% --- Non-user defined constants --- 
k = size(unique(TrainingLabels),1);    % Number of classes
d = size(TrainingData,1);              % Dimension of TrainingData
N = size(TrainingData,2);              % Number of training data

% Initializing one-hot labelmatrix
T = zeros(k,N);

for i = 1:N
    T(TrainingLabels(i),i) = 1;
end

% Initializing matrizes as gaussian noice with 0 mean.
D = normc(2*rand(d,DictionarySize) - 1);
W = normc(2*rand(k,DictionarySize) - 1);

% Concatenating matrices for classification 
TrainingData = [TrainingData; Lambda*T];
D = [D; Lambda*W];

for i = 1:Iterations
    X = OMP(D,TrainingData,Sparcity);
    D = SVDDictionaryUpdate(D,TrainingData,X);

    %Normalizing dictionary part of D matrix
    colnorm = sqrt(sum(D(1:end-k,:).*D(1:end-k,:)));
    D(end-k+1:end,:) = D(end-k+1:end,:)./colnorm;
    D(1:end-k,:) = normc(D(1:end-k,:));   
end

W = D(end-k+1:end,:); 
D = D(1:end-k,:);
end