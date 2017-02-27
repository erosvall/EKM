
% Generating data
close all
D = NormalDataGeneration(randi(10,1),100,2);
scatter(D(:,1),D(:,2))

%% K-means
k = 3;

% Lägger på en hållare för senaste klassen 
ld = [D zeros(size(D,1),1)];
diff = 1;
iteration = 1;

%Generate initial centroids
%c = (1-2*rand(k,2))*max(max(D)); 

c = [];
for p = 1:k
    c = [c ; D(round(rand()*size(D,1)),:)];
end

figure
scatter(D(:,1),D(:,2))
hold on 
scatter(c(:,1),c(:,2),'d')


while(diff > 0.01 && iteration < 100 )
    

    % tilldelar datan närmsta centroid
    for i = 1:size(ld,1)
        temp = ld(i,1:end-1)-c;
        dist = sqrt(sum(temp.*temp,2));
        [a,ind] = min(dist);
        ld(i,3) = ind;
    end
    
    prevc = c;
    %uppdatera centroiderna
    for j = 1:size(c,1)
        choose = ld(:,3)==j;
        d = choose'*ld(:,1:end-1);
        s = nnz(choose);
        if s ~= 0
           c(j,:) = d/s; 
        end
    end
    
    temp = prevc-c;
    diff = max(sqrt(sum(temp.*temp,2)));
    
     iteration = iteration + 1;
end
    

scatter(c(:,1),c(:,2),'*')