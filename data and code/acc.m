%Accuracy calculation for 90:10 data split 
function[nb,nbtrain,nbtest] = acc()
x = xlsread('trainlabels.xlsx');
y = xlsread('testlabels.xlsx');

%model
lda = fitcdiscr(x(:,1),x(:,2));
%predicting train labels
ldaclass = resubPredict(lda);
%LDA train confusion matrix
ldacm = confusionmat(x(:,2),ldaclass);
%trainaccuracy
acc1 = sum(diag(ldacm))/3213 ;
ldatrainacc = acc1*100;
%predicting test labels
labelslda = predict(lda,y(:,1));
%testing confusion matrix
ldacm1 = confusionmat(y(:,2),labelslda);
%test accuracy lda
ldatest = (sum(diag(ldacm1))/357)*100;

%qda model
qda = fitcdiscr(x(:,1),x(:,2),'DiscrimType','quadratic');
%predictig train labels
qdaclass = resubPredict(qda);
%training confusion matrix and accuracy
qdacm = confusionmat(x(:,2),qdaclass);
acc2 = sum(diag(qdacm))/3213 ;
qdatrain= acc2*100;
%predicting test labels, test confusion matrix and test accuracy
labelsqda = predict(qda,y(:,1));
qdacm1 = confusionmat(y(:,2),labelsqda);
qdatest = (sum(diag(qdacm1))/357)*100;


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
