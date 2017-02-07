function [ A ] = OMPfunc( signal, dictionary, sparcity )
%OMPfunc takes the signal to match against and a dictionary
% and returns a sparse Matrix A with the best possible representation
% of the signal. Sparcity gives us the number of atoms per signal in A

% dictionary needs to have normalizied columns 

[n,L] = size(signal);
[m,v] = size(dictionary);

% n = 10; % dim p� data
% L = 2; % antal data
% m = 20; % storlek p� dictionary
s = sparcity; % sparcity

X = signal; % rand(n,L); % data 
D = dictionary; %normc(rand(n,m)); % dictionary med normerade atomer
S = []; % sparce representation av vektor

A = zeros(n,m);

% Steg 1: Sparce coding genom OMP --------------------------

I = []; %Sparar utvalda index

for i = 1:L % f�r varje vektor i data-set
    r = X(:,i); %residualen 
    D0 = []; %Utvalda atomer sparas f�r denna datavektor
    tempI = []; %Utvalda atomers index sparas f�r denna datavektor
    for j = 1:s %
        [maxval, maxarg] = max(abs(r'*D)); %hittar b�sta projektionen
        tempI = [tempI maxarg]; %anv�nda atomer fr�n D
        D0 = [D0 D(:,maxarg)]; %sparar den b�sta atomen fr�n denna iteration  
        a = (D0'*D0)\(D0'*X(:,i)); %projicerar x p� det som sp�ns upp av alla hitils utvalda atomer 
        P = sum(D0*a,2);
        r = X(:,i) - P;
    end

    S = [S a]; % Tempor�r Koefficient vektor 
    I = [I; tempI];
    
    for k = 1:s
        A(I(i,k),i) =  S(k,i);
    end

end

end

