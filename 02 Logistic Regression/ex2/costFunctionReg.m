function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

normalPart = sum(cv)/m; 

thetaSquare = theta(2:end).^2;
regularizePart = (lambda*sum(thetaSquare))/(2*m);

% cost function value
J = normalPart + regularizePart;

%initial gradient descent
thetaT = theta';
X = xtemp;
for i = 1 : v,
  X(:,i) = thetaT(1,i) * X(:,i);
end;

hx = sum(X,2);
hThetaX = sigmoid(hx);
    
difg = hThetaX - y;
dim = size(xtemp,2);

for in = 1 : dim,
    mulg = difg.* xtemp(:,in);
    if in==1,
        grad(in) = (sum(mulg)/m);
    else
        grad(in) = (sum(mulg)/m) + (lambda*theta(in)/m);
    end;
end;
          
%grad = [deltag1/m;deltag2/m;deltag3/m];

% =============================================================

end
