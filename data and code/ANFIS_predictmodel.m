function label = ANFIS_predictmodel(user_input)
%takes in the input from the GUI and runs it on the ANFIS model and 
%gets the crisp output value for the given inputs

%normalising user input to make it suitable to the ANFIS model
%ip= user_input;
ip = cell2mat(user_input);
load data.csv;
A = data;
[m1,m2,norm_value]=norm_data(A,ip);

%reading the ANFIS Model
anfis_model = readfis('ANFIS1');

%predicting the ouptut using the ANFIS Model
norm_output = evalfis(norm_value(1:5),anfis_model);

class_model = classmodel();
label = predict(class_model,norm_output);

end

