function [ labels ] = ELMclassifier(features, inputWeights, outputWeights)
% ELMclassifier.m is given the genereated inputweights and the learnt
% outputweights from ELMtrain.m to label the new featurevectors given.  

    sigma = max(0,inputWeights*features); % RLU on hidden layer inputs. 
    Yres = outputWeights * sigma;
    [~, labels] = max(Yres,[],1);
end

