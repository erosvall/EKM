%% K-SVD Classifier
clear all
close all


trainsize = 200;
valsize = 30;

A = load('MNISTData.mat');
Y = A.imagesTrain(:,1:trainsize);
y = A.labelsTrain(1:trainsize)+1;

Yv =A.imagesTest(:,1:100);
yv = A.labelsTest(1:100)+1;

k = size(unique(y),1);      % antal klasser
d = size(Y,1);              % dimension av data
N = size(Y,2);              % antal data

m = 2*d;                    % storlek av dictionary
lambda = 0.1
sparcity = 20;             % sparcity


accsave = [];

for s = 10:10:100
    Y = A.imagesTrain(:,1:trainsize);
    
    % Initializing one-hot labelmatrix
    T = zeros(k,N);
    for i = 1:N
        T(y(i),i) = 1;
    end

    % Initializing dictionary and 
    D = dictmake(d,m,'U');
    W = dictmake(k,m,'U');
    
    %Initializing sparcerepresentation
    X = zeros(m,N);
    for i = 1:size(X,2)
        for j = 1:sparcity
           X(randi(size(X,1),1),i) = rand(1); 
        end
    end
    
    What = D(end-k+1:end,:);
    That = What*X;
    [~,maxarg] = max(That);

    Trainacc = nnz(maxarg == y')/size(y,1);
    % Concatenating matrices for classification use
    Y = [Y; lambda*T];
    D = [D; lambda*W];


    for i = 1:s
    %    if i == 1
    %        start(s);
    %    end
        X = OMP(D,Y,sparcity);
        D = SVDDictionaryUpdate(D,Y,X);

        %Normalizing D part
        colnorm = sqrt(sum(D(1:end-k,:).*D(1:end-k,:)));
        D(end-k+1:end,:) =  D(end-k+1:end,:)./colnorm;

        D(1:end-k,:) = normc(D(1:end-k,:));
    %    if i == 1;
    %        stop(s);
    %        get(s)
    %    end
        disp(i);
    end

    What = D(end-k+1:end,:);
    That = What*X;
    [~,maxarg] = max(That);

    Trainacc = nnz(maxarg == y')/size(y,1)

    Xtest = OMP(D(1:end-k,:),Yv,sparcity);
    ThatTestYo = What*Xtest;
    [~,testmaxarg] = max(ThatTestYo);
    Testacc = nnz(testmaxarg == yv')/size(yv,1)
    
    accsave = [accsave; Trainacc, Testacc];
end
