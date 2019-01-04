% This script generates and trains ANFIS model using Subclustering ...
% and generates the test, validation and test outputs and errors

load data.csv;
A=data;
length = size(A);

%DATA NORMALISATION
[m1,m2,newdata] = norm_data(A,A);


%TRAINING DATA
train_x = newdata(1:3213,1:length(2)-1);
train_y = newdata(1:3213,length(2));
org_try = data(1:3213,length(2));

% %VALIDATION DATA
% valid_x = newdata(2401:3000,1:length(2)-1);
% valid_y = newdata(2401:3000,length(2));


%TESTING DATA
test_x = newdata(3214:length(1),1:length(2)-1);
test_y = newdata(3214:length(1),length(2));
org_tey = data(3214:length(1), length(2));



%Specifying subclustering gensfis options
opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',0.3,'SquashFactor',1.0);
 
% %Generating Initial FIS Model
fismat1 = genfis(train_x,train_y,opt);
 
%-------------------------------------------------------------------%
%ANFIS

%Initial ANFIS Options
anfisOpt = anfisOptions('InitialFIS',fismat1,'EpochNumber',20,'InitialStepSize',0.1);

%Revised ANFIS Options
%anfisOpt.OptimizationMethod = 0;
anfisOpt.EpochNumber =150;
%anfisOpt.ValidationData = [valid_x,valid_y];
%Suppressing the ANFIS Running Progress Data
anfisOpt.DisplayErrorValues=0;
anfisOpt.DisplayStepSize = 0;
anfisOpt.DisplayFinalResults =0 ;

%Training ANFIS model with training and validation sets
[fismat2,trnerr] = anfis([train_x,train_y],anfisOpt);

%--------------------------------------------------------------------%

%TRAINING CHECKING AND TESTING OUTPUTS
trainop = evalfis(train_x,fismat2);
testop = evalfis(test_x,fismat2); 
%chkop = evalfis(valid_x,fismat2);

%TRAIN CHECK AND TEST ERRORS
trainacc = mse(train_y,trainop);
testacc = mse(test_y,testop);
%RMSEchk = rmse(chkop,valid_y);


