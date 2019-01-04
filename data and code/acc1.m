% Accuracy calculation for data split 80:20
function[nb,nbtrain,nbtest] = acc1()
data = xlsread('data1.xlsx');
x = data(1:2856,:);
y = data(2857:3570,:);


% lda = fitcdiscr(x(:,1),x(:,2));
% ldaclass = resubPredict(lda);
% ldaerr = resubLoss(lda);
% ldacm = confusionmat(x(:,2),ldaclass);
% accu1 = sum(diag(ldacm))/2856 ;
% ldatrain = accu1*100;
% labelslda = predict(lda,y(:,1));
% ldacm1 = confusionmat(y(:,2),labelslda);
% ldatest = (sum(diag(ldacm1))/714)*100;
% 
% qda = fitcdiscr(x(:,1),x(:,2),'DiscrimType','quadratic');
% qdaclass = resubPredict(qda);
% qdacm = confusionmat(x(:,2),qdaclass);
% acc2 = sum(diag(qdacm))/2856 ;
% qdatrain= acc2*100;
% labelsqda = predict(qda,y(:,1));
% qdacm1 = confusionmat(y(:,2),labelsqda);
% qdatest = (sum(diag(qdacm1))/714)*100;



nb = fitcnb(x(:,1),x(:,2));
nbclass = resubPredict(nb);
err = resubLoss(nb);
nbcm = confusionmat(x(:,2),nbclass);
acc3 = sum(diag(nbcm))/2856;
nbtrain=acc3*100;
labelsnb = predict(nb,y(:,1));
nbcm1 = confusionmat(y(:,2),labelsnb);
nbtest = (sum(diag(nbcm1))/714)*100;
end

