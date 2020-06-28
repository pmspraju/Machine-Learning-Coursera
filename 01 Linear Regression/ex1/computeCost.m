function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
v = size(X,2); % number of variants

for i = 1:v,
    thetaT = theta';
    X(:,i) = thetaT(1,i) * X(:,i);
    i = i+1;
end;

h = sum(X,2); % Sum by row wise give the vector having values of 
              % hypothesis function
dif = h - y; %difference between hypothesis value and actual  value
squareError = dif.^2; 

J = sum(squareError)/(2*m); %mean square error or Cost function

% =========================================================================

end
