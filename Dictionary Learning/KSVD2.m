function [ D ] = KSVD2( D, X, A )
%KSVD2 är eriks försökt att följa algoritmen beskriven;
% http://www.ux.uis.no/~karlsk/dle/#ssec32
%Summary of this function goes here
%   Detailed explanation goes here

R = X - D*A;
m = size(D,2);

for i = 1:m
   I = find(A(i,:));
   Ri = R(:,i) + D(:,i)*A(i,I);
   [U,~,~] = svds(Ri,1,'L');
   size(U)
   disp(norm(Ri));
   D(:,i) = U;
   R(:,I) = Ri - D(:,i)*A(i,I);
end


end

