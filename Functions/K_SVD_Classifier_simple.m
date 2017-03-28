%% K-SVD Classifier

clear all
close all

trainsize = 300;
valsize = 100;

A = load('MNISTData.mat');
Y = A.imagesTrain(:,1:trainsize);
y = A.labelsTrain(1:trainsize)+1;
Yv = A.imagesTest(:,1:valsize);
yv = A.labelsTest(1:valsize)+1;

k = size(unique(y),1);      % antal klasser
d = size(Y,1);              % dimension av data
N = size(Y,2);              % antal data

iterations = 20;            % antal uppdateringar av D. 
m = 1960;                  % storlek av dictionary
lambda = 0.4
sparcity = 5;               % sparcity


accsave = [];


Y = A.imagesTrain(:,1:trainsize);

% Initializing one-hot labelmatrix
T = zeros(k,N);
for i = 1:N
    T(y(i),i) = 1;
end

% Initializing dictionary and 
rng(1337);
D = normc(2*rand(d,m) - 1);
rng(1337);
W = normc(2*rand(k,m) - 1);


%Initializing sparcerepresentation
X = zeros(m,N);
for i = 1:size(X,2)
    for j = 1:sparcity
       X(randi(size(X,1),1),i) = rand(1); 
    end
end


% Concatenating matrices for classification use
Y = [Y; lambda*T];
D = [D; lambda*W];

for i = 1:iterations
    X = OMP(D,Y,sparcity);
    D = SVDDictionaryUpdate(D,Y,X);

    %Normalizing D part
    colnorm = sqrt(sum(D(1:end-k,:).*D(1:end-k,:)));
    D(end-k+1:end,:) = D(end-k+1:end,:)./colnorm;
    D(1:end-k,:) = normc(D(1:end-k,:));

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



% -------- Vizualizing the dicpics --------------------
sparcePics = D(1:end-k,:)*X;
%sparcePics = reshape(sparcePics,[28, 28, 200]);
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(sparcePics(:,i),28)');
end

figure
realPics = Y(1:end-10,:);
%realPics = reshape(realPics,[28,28,200]);
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(realPics(:,i),28)');
end

figure
sparceTestPics = D(1:end-k,:)*Xtest;
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(sparceTestPics(:,i),28)');
end

figure
testPics = Yv;
for i = 1:20
    subplot(4,5,i)
    imshow(vec2mat(testPics(:,i),28)');
end












