function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%cost for linear regression
hx = X * theta;
diff = hx - y;
se = diff.^2;
p1 = sum(se)/(2*m);
p21 = theta(2:end).^2;
p2 = (lambda*sum(p21))/(2*m);
J = p1 + p2;

%gradient for linear regression
normalpart = X'*diff;
normalpart = normalpart/m;
regpart1 = normalpart(2:end);
regpart2 = (lambda/m)*theta(2:end);
regpart = regpart1 + regpart2;
grad = [normalpart(1);regpart];
% =========================================================================

grad = grad(:);

end
