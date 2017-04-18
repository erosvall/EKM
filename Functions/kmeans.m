function [ centroids ] = kmeans( dataPoints, numberOfCentroids )
%kmeans Finds N centroids for N assumed unbiased gaussian distributions.
%   Input:
%         dataPoints - Matrix with all data points where each column is an
%             observation.
%         numberOfCentroids - Integer with the assumed number of centroids.
%     Output:
%         c - 
%

% Calculation constants
iteration = 0;
diff = 1;
N = size(dataPoints,2);


% Assign inital centroids, chosen from random datapoints
c = zeros(size(dataPoints,1),numberOfCentroids);
for p = 1:numberOfCentroids
    c(:,p) = dataPoints(:,randi([1,N]));
end

% Class indication:
class = zeros(1,N);

while(diff > 0.01 && iteration < 100 )

    % tilldelar datan närmsta centroid
    for i = 1:size(dataPoints,2)
        temp = dataPoints-c;
        dist = sqrt(sum(temp.*temp,2));
        [~,ind] = min(dist);
        class(1,i) = ind;
    end

    prevc = c;
    %uppdatera centroiderna
    for j = 1:size(c,2)
        choose = class==j;
        d = choose*dataPoints';
        s = nnz(choose);
        if s ~= 0
           c(:,j) = d/s; 
        end
    end

    temp = prevc-c;
    diff = max(sqrt(sum(temp.*temp,2))); % Kolla att vi fortfarande flyttar oss

    iteration = iteration + 1; % Se till att vi inte håller på i all oändlighet.
end

centroids = class;

end

