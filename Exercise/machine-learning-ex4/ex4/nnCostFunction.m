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

%recode labels
y_recode = zeros(num_labels, m) ;
for i=1:m
  y_recode(y(i),i) = 1;
end;

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
A1 = [ones(m,1), X];  %5000x401
Z2 = A1*Theta1';      %5000x25
A2 = sigmoid(Z2);     %5000x25

A2 = [ones(m,1), A2]; %5000x26
Z3 = A2*Theta2';      %5000x10
h = A3 = sigmoid(Z3); %5000x10

kUnitsEachExampleCost = sum(-y_recode.*log(h)' - (1-y_recode).*log(1-h)');
sumAllCost = sum(kUnitsEachExampleCost);

regularization = lambda/(2*m)*(sum(sum(Theta1(:, 2:end).^2)) ...
                             + sum(sum(Theta2(:, 2:end).^2)));
J = 1/m*sumAllCost + regularization;
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
DELTA1 = zeros(size(Theta1));   %25,401
DELTA2 = zeros(size(Theta2));   %10, 26

delta3 = A3' - y_recode;     %10,5000
delta2 = (Theta2(:, 2:end)'*delta3).*sigmoidGradient(Z2');  %(25,10)x(10,5000).*(25,5000)

DELTA1 = delta2*A1;     %(25,5000)x(5000,401)
DELTA2 = delta3*A2;     %(10,5000)x(5000,26)

Theta1_grad = DELTA1/m;     %25,401
Theta2_grad = DELTA2/m;     %10,26
%for i=1:m
%  a1 = A1(i, :)';       %1,401 -> 401, 1
%  z2 = Theta1*a1;       %(25,401)x(401,1)
%  a2 = sigmoid(z2);     %25,1
%  a2 = [1; a2];  %26,1 
%  z3 = Theta2*a2;       %(10,26)x(26,1)
%  a3 = sigmoid(z3);     %10,1
  
%  delta3 = a3 - y_recode(:, i);                            %10,1
%  delta2 = (Theta2(:, 2:end)'*delta3).*sigmoidGradient(z2);  %(25,10)x(10,1).*(25,1)
  
%  DELTA1 = DELTA1 + delta2*a1';     %(25,401) + (25,1)x(1,401)
%  DELTA2 = DELTA2 + delta3*a2';     %(10,26) + (10,1)x(1,26)
  
%  Theta1_grad = DELTA1/m;     %25,401
%  Theta2_grad = DELTA2/m;     %10,26
%end;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:, 2:end) += (lambda/m)*Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda/m)*Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
