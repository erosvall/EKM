function [ D ] = SVDDictionaryUpdate( D, X, A )
%KSVD2 är eriks försökt att följa algoritmen beskriven;
% http://www.ux.uis.no/~karlsk/dle/#ssec32


E = X - D*A; % Create inital Error


for i = 1:size(D,2) % Loop over all columns of the dictionary
    I = find(A(i,:)); % Find all active columns in A
    if ~isempty(I) % If d_i isn't used, ignore that atom
        Er = E(:,I) + D(:,i)*A(i,I); % Choose only the active columns
        [U,S,V] = svds(Er,1,'L');   % Single value decomposition
        D(:,i) = U; % Assign d_k from U
        A(i,I) = S*V'; %Update A, won't affect Sparse representation
        E(:,I) = Er - D(:,i)*A(i,I); % Reset the error matrix
    end
end


end

