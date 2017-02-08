function [ D ] = MOD( X, A )
%MOD Takes Signal and Sparse representation and returns new Dictionary
%   MOD stands for Method of optimal directions and takes 
%   the signal matrix X and sparse representation A and generates a
%   Moore-Penrose pseudoinverse and applies it to signal X.

% MOD w. Moore-Penrose pseudoinverse
%R = (A'*A)\A';
R = A'/(A*A');
%(X*W')/(W*W'); (X*A')/(A*A')
D = normc(X*R);

end

