function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples, e.g. 12

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[r,c]=size(theta); % 2x1
[rx,cx]=size(X); %12x2

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
cost = (X*theta-y)'*(X*theta-y);
newTheta = theta;
newTheta(1) = 0;
reg = newTheta'*newTheta;
J = cost/(2*m) + (lambda*reg)/(2*m) ;

grad(1) = (1/m)*(X*theta-y)'*X(:,1);
for j = 2:size(theta)
  grad(j) = (1/m)*(X*theta-y)'*X(:,j) + lambda*theta(j)/m;
end

% =========================================================================

grad = grad(:);

end
