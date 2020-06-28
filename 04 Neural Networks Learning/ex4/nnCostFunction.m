function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Part 1
% Add ones to the X data matrix
Xplus1 = [ones(m, 1) X];

%get the transpose of theta1
Theta1T = Theta1';

%do a matrix multiplication
hx = Xplus1*Theta1T;

%use sigmoid function to get activation unit values
a2 = sigmoid(hx);

% Add ones to the X data matrix
a2plus1 = [ones(m, 1) a2];

%get the transpose of Theta2
Theta2T = Theta2';

%do a matrix multiplication
hx2 = a2plus1*Theta2T;

%use sigmoid function to get activation unit values
a3 = sigmoid(hx2);

%run a for loop to calculate cost
for i = 1:m
   hx = 0; hxt = 0;
   
   %get output vector for a specific example in training set
   hx = a3(i,:);
   
   %get transpose
   hxt = hx';
   
   yk=0;
   
   %get a vector of zeroes by number of output labels
   yk = zeros(num_labels,1);
   
   %make 1 basing on the actual value of output vector y
   %for example actual value of y is 2, then vetor would be
   %[0 1 0 0 0 0 0 0 0 0] if number of labels is 10
   yk(sub2ind(size(yk),y(i))) = 1;
   
   %calculate the cost function value of current example of training set
   %it would be a vector of size equal to number of output labels
   part1 = ((-1)*yk).*log(hxt);
   part21 = 1-yk;
   part22 = 1-hxt;
   part23 = log(part22);
   part2 = part21.*part23;
   
   yi = part1 - part2;
   
   %sum the values of cost function vector and put it in another vector
   cf(i,:) = sum(yi);
   
end

%final cost function
J = sum(cf)/m;

%theta1 without column corresponding to bias unit(exclude fist column)
Theta1Unbias = Theta1(:,(2:end));
T1ubsquare = Theta1Unbias.^2;
sumByfeatures1 = sum(T1ubsquare,2);
sTheta1 = sum(sumByfeatures1,1);

%theta2 without column corresponding to bias unit(exclude fist column)
Theta2Unbias = Theta2(:,(2:end));
T2ubsquare = Theta2Unbias.^2;
sumByfeatures2 = sum(T2ubsquare,2);
sTheta2 = sum(sumByfeatures2,1);

%cost after regularization
J = J + (lambda/(2*m))*(sTheta1 + sTheta2);

%end of part1

%part2
%Backpropogation to get gradient

% Add ones to the X data matrix
Xplus1 = [ones(m, 1) X];

%for loop to process every element in the training set
GradDelta1 = 0; GradDelta2 = 0;
for t = 1:m
    %get the t-th example from training set
    x_t = Xplus1(t,:);
    a1_t = x_t;
    %get transpose of theta1 vector 
    Theta1T = Theta1';
    
    %calculate hypothesis function for layer 2
    z2_t = x_t*Theta1T;
    a2_t = sigmoid(z2_t);
    
    % Add ones to the a2 data matrix
    a2_t = [1 a2_t];
    
    %get transpose of theta1 vector 
    Theta2T = Theta2';
    
    %calculate hypothesis function for layer 3
    z3_t = a2_t*Theta2T;
    a3_t = sigmoid(z3_t);
    
    ak = a3_t';
    
    %get a vector of zeroes by number of output labels
   yk = zeros(num_labels,1);
   
   %make 1 basing on the actual value of output vector y
   %for example actual value of y is 2, then vetor would be
   %[0 1 0 0 0 0 0 0 0 0] if number of labels is 10
   yk(sub2ind(size(yk),y(t))) = 1;
  
   delta3 = ak - yk;
   
   %calculate delta2
   %dp1 = Theta2(:,2:end)'*delta3;
   %dp21 = a2_t(:,2:end)';
   %dp22 = 1 - dp21;
   %dp2 = dp21.*dp22;
   %delta2 = dp1.*dp2;
   dp1 = (delta3'*Theta2(:,2:end));
   dp2 = sigmoidGradient(z2_t);
   delta2 = dp1.*dp2;
   
   GradDelta1 = GradDelta1 + delta2' * a1_t;
   
   GradDelta2 = GradDelta2 + delta3 * a2_t;
   
end

Theta1_grad_1 = GradDelta1/m;
Theta2_grad_2 = GradDelta2/m;
%end of part2

%part3
Theta1Part = Theta1(:,(2:end));
Theta1Part = (lambda/m)*Theta1Part;

Theta1GradPart = Theta1_grad_1(:,(2:end));
Gradpart11 = Theta1_grad_1((1:end),1);
Gradpart12 = Theta1GradPart + Theta1Part;

Theta1_grad = [Gradpart11 Gradpart12];

Theta2Part = Theta2(:,(2:end));
Theta2Part = (lambda/m)*Theta2Part;

Theta2GradPart = Theta2_grad_2(:,(2:end));
Gradpart21 = Theta2_grad_2((1:end),1);
Gradpart22 = Theta2GradPart + Theta2Part;
Theta2_grad = [Gradpart21 Gradpart22];
%end of part3
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
