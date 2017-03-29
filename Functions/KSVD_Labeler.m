function [Label,SparseData] = KSVD_Labeler(Data,Dictionary,Weights,Sparcity)
% Data is a d x N matrix containg the N datapoints. The dicitionary is used
% to find the sparce represenation X, using only "Sparcity" number of atoms
% from the dictionary. 


% Calculating the sparce representation for the given data
X = OMP(Dictionary,Data,Sparcity);

% Calculating the labelmatrix
T  = Weights*X;

% Classifing data 
[~,Label] = max(T);

% Generating the Sparce data for visualization
SparseData = Dictionary*X;
return 