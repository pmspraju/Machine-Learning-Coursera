function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
v = size(X,2); % number of features. in assignment it is 3 after adding x0
thetaT = theta'; %theta transpose
for i = 1 : v,
    X(:,i) = thetaT(1,i) * X(:,i);
end;

% theta0*x0 + theta1*x1 + theta2*x2
hx = sum(X,2);

%feed above matrix to sigmoid function, to get sigmoid value if each value
hThetaX = sigmoid(hx);

pos = find(hThetaX >= 0.5);
neg = find(hThetaX < 0.5);

p(pos) = 1;
p(neg) = 0;

% =========================================================================


end
