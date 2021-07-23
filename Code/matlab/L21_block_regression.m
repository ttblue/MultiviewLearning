function U1 = L21_block_regression(Y,X,lambda,tol,max_iterations,verbose)
% input: Y - matrix, each column is a function
%        X - cell array of feature matrices
%        lambda - regularization strength
%        tol - stoping criterion
%        max_iterations - maximum number of iterations
% output: W 
% Minimizes sum_i |Y(:,i)-X{i}*W(:,i)|_{2}^2 + lambda*|W|_{2,1}
%
% algorithm adapted from:
% @inproceedings{nie2010efficient,
%   title={Efficient and robust feature selection via joint ℓ2, 1-norms minimization},
%   author={Nie, Feiping and Huang, Heng and Cai, Xiao and Ding, Chris H},
%   booktitle={Advances in neural information processing systems},
%   pages={1813--1821},
%   year={2010}
% }
%

U1 = 0;

if nargin<3
  disp(['ERROR: not enough arguments L21_block_regression(Y,X,lambda,...).']);
  return;
end
if nargin<4, tol=1e-6; end;
if nargin<5, max_iterations = 100; end;
if nargin<6, verbose = 0; end;
if ~strcmp(class(Y),'double')
  disp(strcat(['ERROR: Invalid class for Y (',class(Y),'), expected class double.']));
  return;
end
if ~strcmp(class(X),'cell')
  disp(strcat(['ERROR: Invalid class for X (',class(X),'), expected class cell.']));
  return;
end
if length(X)~=size(Y,2) && length(X)~=1
  disp(strcat(['ERROR: length of cell array X should be 1 or size(Y,2).']));
  return;
end
for i=1:length(X)
if ~strcmp(class(X{i}),'double')
  disp(strcat(['ERROR: Invalid class for X{',num2str(i),'} (',class(X{i}),'), expected class double.']));
  return;
end
if size(X{i},1)~=size(Y,1)
  disp(strcat(['ERROR: Number of rows not consistent for Y and X{',num2str(i),'}.']));
  return;
end
if size(X{i},2)~=size(X{1},2)
  disp(strcat(['ERROR: Number of rows not consistent for X{',num2str(i),'} and X{1}.']));
  return;
end
end

[n,d] = size(Y);
T = length(X);
[~,m] = size(X{1});
last_objective_value = inf;
D = spdiags(ones(m,1),0,m,m);
if m<n
  L = spdiags(lambda.^2*ones(m,1),0,m,m);
  XX = {}; XY = {};
  for i=1:T, XX{i} = X{i}'*X{i}; end;
  for j=1:d
    jdx = mod(j-1,T)+1;
    XY{j} = X{jdx}'*Y(:,j); 
  end;
else, L = spdiags(lambda.^2*ones(n,1),0,n,n); end;
U1 = zeros(m,d);
U2 = zeros(n,d);

for it=1:max_iterations
  for j=1:d
    jdx = mod(j-1,T)+1;
    if m<n, v = (1/lambda.^2)*( Y(:,j)-X{jdx}*( (L+D*XX{jdx})\(D*XY{j}) ) );
    else, v = (L+X{jdx}*D*X{jdx}')\Y(:,j); end;
    U1(:,j) = D*X{jdx}'*v;
    U2(:,j) = lambda*v;
  end
  D = spdiags(2*sqrt(sum(U1.^2,2)),0,m,m);
  objective_value = sum(nonzeros(D));
  delta = last_objective_value-objective_value;
  if isnan(delta), break; end;
  last_objective_value = objective_value;
  if delta < tol, break; end;
  if verbose, disp(strcat([num2str(it),': ',num2str(delta)])); end;
end

if isnan(delta), disp('ERROR: something is terribly wrong here.'); end;
if it==max_iterations, disp(strcat(['WARNING: max (',num2str(max_iterations),') iterations reached with tol: ',num2str(tol),'. Final delta: ',num2str(delta)])); end;
