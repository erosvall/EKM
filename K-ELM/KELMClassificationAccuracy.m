function [Accuracy] = KELMClassificationAccuracy(TrainFeatures,TrainLabels,TestFeatures,TestLabels,lambda,kernel,kernelparam)
    

    [wi, wo, sigma] = KELMtrainer(TrainFeatures,TrainLabels,lambda,kernel,kernelparam);    
    labelGuess = KELMclassifier(TestFeatures,sigma,wi,wo,kernel,kernelparam);
    Accuracy = nnz(labelGuess == TestLabels)/size(testL,2);
end