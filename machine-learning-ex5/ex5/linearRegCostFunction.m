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

n = length(theta);
% Add intercept term to x and X_test
h = X*theta;
cost = h-y;
cost_sq = cost.*cost;
ads=sum(cost_sq,1);
J_unreg = (1/(2*m))*(ads);
thetaT = theta;
thetaT(1) = 0;
theta_reg = thetaT.*thetaT;
J_reg   =  (lambda/(2*m))*sum(theta_reg,1);
J = J_unreg + J_reg ;

grad = 1 / m * X' * (h - y);
grad(2:n) = grad(2:n) + lambda / m * theta(2:n);









% =========================================================================

grad = grad(:);

end
