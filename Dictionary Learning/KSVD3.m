
sparcity  = 3; % sparcit

n = 5; % dimension av data
p = 10; % antal data
m = 15; % storlek av dictionary

D0 = normc(rand(n,m)); % Det dictionary vi försöker återskapa så bra som möjligt
D =  normc(rand(n,m)); % Initierings D för algoritm 

A0 = zeros(m,p); % Sparce-representationsmatris
for i = 1:size(A0,2)
    for j = 1:sparcity
       A0(randi(size(A0,1),1),i) = rand(1); 
    end
end

A = zeros(m,p); % behöver en annan matris att skicka till k-svd
for i = 1:size(A,2)
    for j = 1:sparcity
       A(randi(size(A,1),1),i) = rand(1); 
    end
end

X = D0*A0; % Data

%%
temp = D*A;

for i = 1:1
   I = find(A(i,:));
   E = X(:,I) - temp(:,I) +  
   
   Ri = R(:,i) + D(:,i)*A(i,I);
   [U,~,~] = svds(Ri,1,'L');
   size(U)
   disp(norm(Ri));
   D(:,i) = U;
   R(:,I) = Ri - D(:,i)*A(i,I);
end




