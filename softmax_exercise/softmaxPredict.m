function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.



numClasses = size(theta ,1);  
numCases = size(data ,2);  
M = zeros(numClasses,numCases);  
M = theta*data;  
[~,pred] = max(M ,[] ,1);  

% M = theta*data;
% [Y,pred] = max(M,[],1);

% ---------------------------------------------------------------------

end

