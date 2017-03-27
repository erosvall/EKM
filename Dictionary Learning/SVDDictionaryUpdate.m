function [ D ] = SVDDictionaryUpdate( D, Y, X )
%KSVD2 är eriks försökt att följa algoritmen beskriven;
% http://www.ux.uis.no/~karlsk/dle/#ssec32
% D dictionary, Y data, X sparse represenattion 

E = Y - D*X; % Create inital Error

for i = 1:size(D,2)                     % Loop over all columns of the dictionary
    I = find(X(i,:));                   % Find all active columns in A
    if ~isempty(I)                      % If d_i isn't used, ignore that atom
        Er = E(:,I) + D(:,i)*X(i,I);    % Choose only the active columns
        [U,S,V] = svds(Er,1,'L');       % Single value decomposition
        D(:,i) = U;                     % Assign d_k from U
        X(i,I) = S*V';                  % Update A, won't affect Sparse representation
        E(:,I) = Er - D(:,i)*X(i,I);    % Reset the error matrix
    end
end
end

