%% Viktors försök på en OMP + K-SVD dictionary learning algoritm
n = 10; % dim på data
L = 2; % antal data
m = 20; % storlek på dictionary
s = 4; % sparcity

X = rand(n,L); % data 
D = normc(rand(n,m)); % dictionary med normerade atomer
A = []; % sparce representation av vektor



% Steg 1: Sparce coding genom OMP --------------------------

I = []; %Sparar utvalda index

for i = 1:L % för varje vektor i data-set
    r = X(:,i); %residualen 
    D0 = []; %Utvalda atomer sparas för denna datavektor
    tempI = []; %Utvalda atomers index sparas för denna datavektor
    for j = 1:s %
        [maxval, maxarg] = max(abs(r'*D)); %hittar bästa projektionen
        tempI = [tempI maxarg]; %använda atomer från D
        D0 = [D0 D(:,maxarg)]; %sparar den bästa atomen från denna iteration  
        a = (D0'*D0)\(D0'*X(:,i)); %projicerar x på det som späns upp av alla hitils utvalda atomer 
        P = sum(D0*a,2);
        r = X(:,i) - P;
        norm(r) % Vad gör denna? Sparar den r som normerad?
    end
    
    A = [A a]; % Koefficienterna 
    I = [I; tempI]
    
end

