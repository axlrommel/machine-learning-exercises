function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

y_matrix= eye(num_labels)(y,:); %size 5000x10
a1 = [ones(size(X, 1), 1) X]; % add column of 1s
z2 = a1 * Theta1';
atemp2 = sigmoid(z2);
a2 = [ones(size(atemp2, 1), 1) atemp2]; %add column of 1s
z3 = a2 * Theta2';
a3 = sigmoid(z3); %size 5000x10

for i = 1:m
  for k = 1:num_labels 
     J = J - (y_matrix(i,k)*log(a3(i,k)))-((1-y_matrix(i,k))*(log(1 - a3(i,k)))); 
  end
end

J = J/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for t= 1:m
  a1g = [ones(1,1) X(t,:)]; %1x401
  z2g = a1g * Theta1'; %1x25
  a2g = [ones(1,1) sigmoid(z2g)]; %1x26
  z3g = a2g * Theta2'; %1x10
  a3g = sigmoid(z3g); %1x10;
  d3g = a3g - y_matrix(t,:); %1x10
  d2g = (d3g*(Theta2));
  d2g = d2g(2:end); % from a row vector remove the first element
  d2g = d2g .* sigmoidGradient(z2g);
  Theta2_grad = Theta2_grad + (a2g'*d3g)';
  Theta1_grad = Theta1_grad + (a1g'*d2g)';
end
  temp2 = Theta2;
  temp2(:,1)=0; %make theta(0) = 0
  temp1 = Theta1;
  temp1(:,1)=0; %make theta(0) = 0
  Theta2_grad = Theta2_grad/m + (lambda*temp2)/m;
  Theta1_grad = Theta1_grad/m + (lambda*temp1)/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

newTheta1 = Theta1; %size: hidden_layer_size, input_layer_size
newTheta2 = Theta2; %size: num_labels, hidden_layer_size
newTheta1(:,[1])=[]; % remove the first column
newTheta2(:,[1])=[]; % remove the first column
reg = 0;
for j=1:hidden_layer_size
  for k=1:input_layer_size
    reg = reg + (newTheta1(j,k))^2;
  end
end

for j=1:num_labels
  for k=1:hidden_layer_size
    reg = reg + (newTheta2(j,k))^2;
  end
end
reg = reg*lambda/(2*m);

J = J + reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
