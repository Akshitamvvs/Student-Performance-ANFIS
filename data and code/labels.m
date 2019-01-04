%Assigning labels to training and test data
trainlabels=[];
for i = 1:size(org_try)
    if org_try(i)<=59.9
        trainlabels(i) = 3;
    elseif (org_try(i)>=60 && org_try(i)<=79.9)
            trainlabels(i) =2;
    else
        trainlabels(i)=1;
    end
end
trainlabels = trainlabels';

testlabels=[];
for i = 1:size(org_tey)
    if org_tey(i)<=59.9
        testlabels(i) = 3;
    elseif (org_tey(i)>=60 && org_tey(i)<=79.9)
            testlabels(i) =2;
    else
        testlabels(i)=1;
    end
end
testlabels = testlabels';
       
            