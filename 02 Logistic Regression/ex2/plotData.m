function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
fprintf('Plotting Data ...\n')
data = load('ex2data1.txt'); % read comma separated data 
X1 = data(:, 1); X2 = data(:, 2); %assign entire first column to X and entire second column to X2
y = data(:, 3); % output column to y
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m

%below plot will plot both positive and negative with same cross(x)
%plot(X1,X2,'rx','MarkerSize',10); % plots X versus Y with cross(x) marks red in color of size 10

%so to plot separately, we will create two matrices with indices of 
%pos ->for positive(y==1) and neg -> for negative (y==0)
pos = find(y==1);
neg = find(y==0);

%now plot pos first
plot(X1(pos),X2(pos), 'g+','MarkerSize',7);

%now plot pos first
plot(X1(neg),X2(neg), 'ro', 'MarkerFaceColor', 'y', 'MarkerSize',7);

legend('Admitted','Not Admitted');

xlabel('Exam 1 score');
ylabel('Exam 2 score');
fprintf('Program paused. Press enter to continue.\n');
% =========================================================================



hold off;

end
