function [mse] = untitled2(X1,X2)
% calculates the mean square error
l = size(X1);
% X1 -- predicted output
% X2 -- original output
mse =sum((X1-X2).^2)/l(1);

end