function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
x=data;
z=theta*x;
z=bsxfun(@minus,z,max(z,[],1));
z=exp(z);
z=bsxfun(@rdivide,z,sum(z));
cost=-1/numCases*sum(sum((groundTruth.*log(z))))+lambda/2*sum(sum(theta.^2));

thetagrad=(-1/numCases*x*(groundTruth'-z')+lambda*theta')';

% M = theta*data;
% NorM = bsxfun(@minus, M, max(M, [], 1));  %归一化，每列减去此列的最大值，使得M的每个元素不至于太大。
% ExpM = exp(NorM);
% P = bsxfun(@rdivide,ExpM,sum(ExpM));      %概率
% cost = -1/numClasses*(groundTruth(:)'*log(P(:)))+lambda/2*(theta(:)'*theta(:)); %代价函数
% thetagrad =  -1/numClasses*((groundTruth-P)*data')+lambda*theta;       %梯度    








% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

