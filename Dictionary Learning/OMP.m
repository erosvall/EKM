function [ A ] = OMP(dictionary, signal, sparcity)
%OMPfunc takes the signal to match against and a dictionary
% and returns a sparse Matrix A with the best possible representation
% of the signal. Sparcity gives us the number of atoms per signal in A

% dictionary needs to have normalizied columns 

p = size(signal,2);
m = size(dictionary,2);
s = sparcity; % sparcity

X = signal; % data 
D = dictionary; %normc(rand(n,m)); % dictionary med normerade atomer

A = zeros(m,p);

for i = 1:p % f�r varje vektor i data-set
    r = X(:,i); %residualen 
    D0 = []; %Utvalda atomer sparas f�r denna datavektor
    for j = 1:s %
        [maxval, maxarg] = max(abs(r'*D)); %hittar b�sta projektionen
        D0 = [D0 D(:,maxarg)]; %sparar den b�sta atomen fr�n denna iteration  
        a = (D0'*D0)\(D0'*X(:,i)); %projicerar x p� det som sp�ns upp av alla hitils utvalda atomer 
        P = sum(D0*a,2);
        r = X(:,i) - P;
        A(maxarg,i) = a(j,1);
    end
end

end

