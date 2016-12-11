function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
a = size(X_norm);
for i = 1:size(X_norm,2);
	mu(:,i) = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));
end
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
a_r=a(1,1);
a_c=a(1,2);
for i=1:a_c;
	mu_1 = mean(X_norm(:,i));
	mu(:,i)=mu_1;
end

for i=1:a_c;
	stdv_x1 = std(X_norm(:,i));
	sigma(:,i)=stdv_x1;
end

for i=1:a_c;
	X_norm(:,i)=X_norm(:,i) - mu(1,i);

end

for i=1:a_c;
	X_norm(:,i)=X_norm(:,i)/sigma(1,i);

 
end




% ============================================================
