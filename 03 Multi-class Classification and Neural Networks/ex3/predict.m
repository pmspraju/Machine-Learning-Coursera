function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

%multiply X with transpose of Theta1 to get matrix of activation unit
%values
a2values = Theta1*X';
a2values = sigmoid(a2values);

% Add ones to the activation layer data matrix
a2values = [ones(1, m); a2values];

%multiply activation unit values with transpose of Theta2 to get hypothesis
%function values h(x)
hx = Theta2*a2values;
hx = sigmoid(hx);
[v,i] = max(hx,[],1);

p = i';


% =========================================================================


end
