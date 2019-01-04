function [M1,M2,newdata] = norm_data(A,arr)

M1= [max(A(:,1)) max(A(:,2)) max(A(:,3)) max(A(:,4)) max(A(:,5)) max(A(:,6))];

%min of each col
M2= [min(A(:,1)) min(A(:,2)) min(A(:,3)) min(A(:,4)) min(A(:,5)) min(A(:,6))];

for i = 1:6
   newdata(:,i) = abs(arr(:,i)-M2(i))/abs(M1(i)-M2(i));
end
end