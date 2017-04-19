function [Accuracy] = KELMClassificationAccuracy(TrainFeatures,TrainLabels,TestFeatures,TestLabels,lambda,hiddenNodes,kernel,kernelparam)
    

    [wi, wo, sigma] = KELMtrainer(TrainFeatures,TrainLabels,hiddenNodes,lambda,kernel,kernelparam);    
    labelGuess = KELMclassifier(TestFeatures,sigma,wi,wo,kernel,kernelparam);
    labelGuess(1:10)
    TestLabels(1:10)
    Accuracy = nnz(labelGuess == TestLabels)/size(labelGuess,2);
end