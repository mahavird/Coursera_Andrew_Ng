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


z = X*theta;
h = sigmoid(z);
l = log(h);
pl = -y'*l;
pr=-(1-y)'*log(1-h);
p=pl+pr;


j1 = (1/m)*(sum(p,1));

thetaT = theta;
thetaT(1) = 0; %making the first element of thetat vector Zero to avoid error caused by "1" 

j2=0;
len=size(thetaT,1);
thetasq=thetaT.*thetaT;
thetasum=sum(thetasq,1);
j2=(lambda/(2*m))*thetasum;

J=j1+j2;

%grad = (1/m)*(X'*(h-y));
grad = (X' * (sigmoid(X * theta) - y)) * (1/m) + thetaT * (lambda / m);



% =============================================================

end
