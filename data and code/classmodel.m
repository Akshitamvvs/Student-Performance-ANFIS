function [nb,nbtrain,nbtest] = classmodel()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
x = xlsread('trainlabels.xlsx');
y = xlsread('testlabels.xlsx');
%naives bayes model training
nb = fitcnb(x(:,1),x(:,2));
%predicting train labels, confusion matrix and train accuracy
nbclass = resubPredict(nb);
err = resubLoss(nb);
nbcm = confusionmat(x(:,2),nbclass);
acc3 = sum(diag(nbcm))/3213;
nbtrain=acc3*100;
%predicting test labels, test confusion matrix and test accuracy
labelsnb = predict(nb,y(:,1));
nbcm1 = confusionmat(y(:,2),labelsnb);
nbtest = (sum(diag(nbcm1))/357)*100;

end

