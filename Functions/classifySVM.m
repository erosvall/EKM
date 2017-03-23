function [ res ] = classifySVM( newV, alphas, arg, p )
%classifySVM takes a matrix and returns the class of each vector
%   
%     Input:
%         newV - NxM matrix where N is the dimension of the data and M is the
%         number of items to be classified.
%         alphas - Our SVM. KxBxD where K is the number of classes, B the number
%         of support vectors and D the information associated with that 
%         support vector.
%         arg - Choice of Kernel, valid inputs are 'lin', 'rad' and 'pol'
%         p - Kernel Parameter, contextual. If kernel is radial it's sigma. 
%         If the kernel is polynomial it's p.
% 
%     Output:
%         res - 1xM vector with the classes of all the input vectors.






memory = zeros(size(alphas,1),size(newV,2)); % A NxM matrix where N is the number of vectors in newV and M is the number of classes
switch arg
    case {'lin'}
        for k = 1:size(newV,2)
            for j = 1:size(alphas,1)
                v = 0;
                for i = 1:size(alphas,2)                     
                    v = v + alphas(j,i,1)*alphas(j,i,end)*linKerl(newV(:,k)', reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ) ) ;                
                end
                memory(j,k) = v;        
            end
        end
        
    case {'rad'}
        for k = 1:size(newV,2)
            for j = 1:size(alphas,1)
                v = 0;
                for i = 1:size(alphas,2)
                    v = v + alphas(j,i,1)*alphas(j,i,end)*radKerl(newV(:,k), reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ), p) ;
                end    
                memory(j,k) = v;            
            end
        end
        
    case {'pol'}
        for k = 1:size(newV,2)
            for j = 1:size(alphas,1)
                v = 0;
                for i = 1:size(alphas,2)
                    v = v + alphas(j,i,1)*alphas(j,i,end)*polyKerl(newV(:,k)', reshape( alphas(j,i,2:end-1) , [size(alphas(j,i,2:end-1),3),1] ) , p);
                end   
                memory(j,k) = v;

            end
        end
        
end

[~, res] = max(memory);

end
