function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
thetaSize = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

cost = 0;
for i = 1 : m
   expnt = X(i,:)*theta;
   cost = cost + (y(i)*log(sigmoid(expnt))) + (1 - y(i))*(log(1 - sigmoid(expnt)));
end
reg = 0;
for j = 2: thetaSize
 reg = reg + theta(j)^2;  
end
J = -cost/m + (lambda*reg)/(2*m) ;

for j = 1 : thetaSize
  derivJ = 0;
  for i = 1 : m
     expnt = (theta)'*X(i,:)';
     derivJ = derivJ + (sigmoid(expnt) - y(i))*X(i,j);
  end
  grad(j) = derivJ/m;
  if(j > 1)
    grad(j) = grad(j) + lambda*theta(j)/m; 
end
end
