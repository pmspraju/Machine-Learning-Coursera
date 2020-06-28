function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%theta = [-1.148717;0.569809;0.111394];
xtemp = X;
v = size(X,2); % number of features. in assignment it is 3 after adding x0
thetaT = theta'; %theta transpose

% theta0*x0, theta1*x1, theta2*x2
for i = 1 : v,
    X(:,i) = thetaT(1,i) * X(:,i);
end;

% theta0*x0 + theta1*x1 + theta2*x2
hx = sum(X,2);

%feed above matrix to sigmoid function, to get sigmoid value if each value
hThetaX = sigmoid(hx);

v1 = (-1).*y;
v2 = log(hThetaX);
v3 = (1 - y);
v41 = (1 - hThetaX);
v4 = log(v41);
cv = ((v1.*v2) - (v3.*v4));

% cost function value
J = sum(cv)/m;

%Gradient decent 
num_iters = 1500; alpha = 0.01;
%for iter = 1 : num_iters,
    
    thetaT = theta';
    X = xtemp;
    for i = 1 : v,
    X(:,i) = thetaT(1,i) * X(:,i);
    end;
    
    hx = sum(X,2);
    hThetaX = sigmoid(hx);
    
    difg = hThetaX - y;
    
    mulg = difg.* xtemp(:,1);
    deltag1 = sum(mulg);
    
    mulg = difg.* xtemp(:,2);
    deltag2 = sum(mulg);
    
    mulg = difg.* xtemp(:,3);
    deltag3 = sum(mulg);
      
    %theta1 = theta(1) - (1/m)*(alpha * deltag1);
    %theta2 = theta(2) - (1/m)*(alpha * deltag2);
    %theta3 = theta(3) - (1/m)*(alpha * deltag3);
    %theta = [theta1;theta2;theta3];
    
    grad = [deltag1/m;deltag2/m;deltag3/m];

%end

% =============================================================

end
