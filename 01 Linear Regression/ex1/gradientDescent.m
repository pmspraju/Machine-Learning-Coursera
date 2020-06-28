function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

v = size(X,2); % number of variants

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    Xg = X; yg = y;
    thetaT = theta';
    for i = 1:v,        
        Xg(:,i) = thetaT(1,i) * Xg(:,i);
        %i = i+1;
    end
    
    hg = sum(Xg,2);
    difg = hg - yg;
    
    mulg = difg.* X(:,1);
    deltag1 = sum(mulg);
    
    mulg = difg.* X(:,2);
    deltag2 = sum(mulg);
      
    theta1 = theta(1) - (1/m)*(alpha * deltag1);
    theta2 = theta(2) - (1/m)*(alpha * deltag2);
    theta = [theta1;theta2];
        
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
