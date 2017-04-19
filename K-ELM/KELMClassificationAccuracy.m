function [Accuracy] = KELMClassificationAccuracy(TrainFeatures,TrainLabels,TestFeatures,TestLabels,lambda,hiddenNodes,kernel,kernelparam)
    

    [wi, wo, sigma] = KELMtrainer(TrainFeatures,TrainLabels,hiddenNodes,lambda,kernel,kernelparam);    
    labelGuess = KELMclassifier(TestFeatures,sigma,wi,wo,kernel,kernelparam);
    if size(TestLabels,1) > 1
        [~,TestLabels] = max(TestLabels);
    end
    
    Accuracy = nnz(labelGuess == TestLabels)/size(labelGuess,2);
end