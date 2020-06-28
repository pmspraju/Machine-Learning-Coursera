function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
val = [0.01 0.03 0.1 0.3 1 3 10 30];
%find C and sigma having minimum error
for i = 1:8
    C_t = val(i);
    for j = 1:8
        sigma_t = val(j);
        %train model using temp C and sigma
        model= svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t));
        
        %using above calculated model, using cross validation set, get
        %predictions of y
        predictions = svmPredict(model, Xval);
        
        %calculate error between above predicted output and actual yval
        error = mean(double(predictions ~= yval));
        
        %for first time error will be hold-error
        if(i==1 && j==1)
            hold_error = error;
        else
            %else if found error is less than previous held error, then
            %change hold error and store C and sigma, as it has min error
            if (error < hold_error)
                hold_error = error;
                C = C_t;
                sigma = sigma_t;
            end;
        end;
    end;
end;

% =========================================================================

end
