%% Viktors f�rs�k p� en OMP + K-SVD dictionary learning algoritm
n = 10; % dim p� data
L = 2; % antal data
m = 20; % storlek p� dictionary
s = 4; % sparcity

X = rand(n,L); % data 
D = normc(rand(n,m)); % dictionary med normerade atomer
A = []; % sparce representation av vektor



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
        norm(r) % Vad g�r denna? Sparar den r som normerad?
    end
    
    A = [A a]; % Koefficienterna 
    I = [I; tempI]
    
end

