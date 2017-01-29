function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(X, 2);
fprintf('\No of Input (training Examples) to NN %i\n',m);
fprintf('\Size of features of every Training Example to NN %i\n',n);
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

fprintf('\Inside Neural network predict function\n');

X = [ones(m, 1) X];     %adds 1 to all the rows of X (bias unit)

a2 = sigmoid(X * Theta1');

m1 = size(a2, 1);
n1 = size(a2, 2);
fprintf('\No of rows in a2 %i\n',m1);
fprintf('\ No of cols in a2 %i\n',n1)

a2 = [ones(m, 1) a2];   %adds 1 to all the rows of a2
htheta = sigmoid(a2 * Theta2');


m2 = size(htheta, 1);
n2 = size(htheta, 2);
fprintf('\No of rows in htheta %i\n',m2);
fprintf('\ No of cols in htheta %i\n',n2)

printf("%i\n", htheta);

[temp, p] = max(htheta, [], 2); % Returns Max value among cols of each row of htheta matrix









% =========================================================================


end
