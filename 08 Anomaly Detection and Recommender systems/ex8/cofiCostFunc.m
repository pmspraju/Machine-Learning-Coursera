function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
y_predict = X * Theta';
diff = y_predict - Y;
sqrErr = diff.^2;
sum_r = sum(sqrErr(R == 1));
J = (1/2) * sum_r + ((sum(sum((Theta.^2),2),1))*(lambda/2)) + ...
                    ((sum(sum((X.^2),2),1))*(lambda/2));

%gradient for features
for i = 1:num_movies
    %users who rated i'th movie
    idx = find(R(i,:) == 1);
    %size(idx)
    %theta matrix for above users (who rated i'th move)
    Theta_temp = Theta(idx,:);
    %size(Theta_temp)
    %actual ratings for the i'th movie - only users who rated the movie
    Y_temp = Y(i,idx);
    %size(Y_temp)
    %predicted rating for i'th move for which user has  rated
    Y_predict = X(i,:) * Theta_temp';
    %size(Y_predict)
    
    diff = Y_predict - Y_temp;
    
    X_grad(i,:) = (diff * Theta_temp) + ((lambda) * X(i,:));
    
end;

%gradient for theta
for i = 1:num_users
    %users who rated i'th movie
    idx = find(R(:,i) == 1);
    
    %theta matrix for above users (who rated i'th move)
    Theta_temp = Theta(i,:);
    
    %actual ratings for the i'th movie - only users who rated the movie
    Y_temp = Y(idx,i);
    
    %predicted rating for i'th move for which user has  rated
    Y_predict = X(idx,:) * Theta(i,:)';
    
    diff = Y_predict - Y_temp;
    %size(diff)
    %size(X(idx,:))
    Theta_grad(i,:) = (diff' * X(idx,:)) + ((lambda) * Theta(i,:));
end;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
