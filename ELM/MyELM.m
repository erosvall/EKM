% Pleb way of ELM
%% Loading files
imagesTrain = loadMNISTImages('train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels.idx1-ubyte');

imagesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% functions
rlu = @(w,x,b) max(0,w*x+b);
sigmoid = @(w,x,b) 1./(1+exp((w*x + b)));
%% Training!
N = 6000; %number of datapoints

h = 1000; % number of hidden layers

X = imagesTrain(1:end,1:N);         % Smaller feature matrix 
sublables = labelsTrain(1:N,1);     % correspoinding lables

B = ones(h,size(X,2));              % Initialize bias term
C = 2*rand(1,N)-1;

Wr = dictmake( h, size(X,1));       % random weights [-1,1]

sigma = sigmoid(Wr,X,B);            % sigmoid function (is this kernel?)
         

Y = zeros(10,size(X,2));            % Initialize class matrix

% Fill class matrix
for i = 1:size(sublables,1)
    Y(sublables(i,1)+1,i) = 1;
end

W1 = Y*pinv(sigma);                 % Calculated weiths in output layer

%% Testing
Xtest = imagesTest;
Ytest = labelsTest';
Btest = ones(h,size(Xtest,2));

sigmatest = sigmoid(Wr,Xtest,Btest);

Yres = W1 * sigmatest;
[maxval, maxind] = max(Yres,[],1);
maxind = maxind-1;

accuracy = nnz(Ytest == maxind)/size(Ytest,2)

