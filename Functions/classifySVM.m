function [ res ] = classifySVM( newV, alphas, arg )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

memory = zeros(size(alphas,1),size(newV,2)); % A NxM matrix where N is the number of vectors in newV and M is the number of classes
switch arg
    case {'lin'}
        for j = 1:size(alphas,1)
            v = 0;
            for i = 1:size(alphas,2)
                v = v + alphas(j,i,1)*alphas(j,i,end)*linKerl(newV', reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ) ) ;                
            end
            memory(j) = v;         
        end
        
        
    case {'rad'}
        for j = 1:size(alphas,1)
            v = 0;
            for i = 1:size(alphas,1)
                v = v + alphas(j,i,1)*alphas(j,i,end)*radKerl(newV, reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ), 20) ;
            end    
            memory(j) = v;            
        end

        
    case {'pol'}
        for k = 1:size(newV,2)
            for j = 1:size(alphas,1)
                v = 0;
                for i = 1:size(alphas,1)
                    v = v + alphas(j,i,1)*alphas(j,i,end)*polyKerl(newV(:,k)', reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ) ,2);
                end   
                memory(j,k) = v;

            end
        end
        
end


[~, res] = max(memory);

end
