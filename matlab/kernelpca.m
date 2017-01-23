addpath('mi');

%===========================================
% get observed data
%  pig bleed
% 1 - time
% 2 - EKG     
% 3 - Art pressure MILLAR
% 4 - Art pressure Fluid Filled
% 5 - Pulmonary pressure
% 6 - CVP
% 7 - Plethysmograph
% 8 - CCO
% 9 - SVO2
% 10 -SPO2
% 11 - Airway pressure
% ann = strsplit(fileread('33_annotation.txt'),'\n');
% ann_idx=[]; ann_text={};k=1;
% for i=1:length(ann)
%   s = ann{i};
%   idx = find(isspace(s));
%   if length(idx)==0, continue; end;
%   ann_idx(k) = str2num(s(1:idx(1)));
%   ann_text{k} = s(idx(1)+1:end);
%   k=k+1;
% end
% ann_use = [3,8; 9,14; 19,20; 27,28; 31,36; 41,41; 44,47; 48,49; 52,53; 59,59; 60,60; 61,61];
data_file = '/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/slow/33.csv';
header = 1;
start = 1;
nrow = 5484735;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread(data_file,'\t',datrows);
y = X(:,4);
y = tsmovavg(y,'t',5,1); y = y(~isnan(y)); 
plot(y)

%===========================================
% Average Mutual Information & embedding
K = 100;
y0 = y(1:5000);
ami = zeros(K-1,1);
tau=0; fmv=inf;
for k=1:K-1
 ami(k) = mutualinfo(y0(1:end-k)',y0(1+k:end)');
 if ami(k)<=fmv
   tau = k; fmv=ami(k);
 else
   fmv=-inf;
 end
end
plot(ami)
tau

D = 3;
x = [];
for d=1:D
  x = [x(1:end-tau,:),y((d-1)*tau+1:end)];
end
L = 5000; % length of sample
d = 1;
plot3(x(1:d:L,1),x(1:d:L,2),x(1:d:L,3))
%===========================================
% perform analysis

D = 6; % number of dimensions to reduce to
N = 100; % number of samples
a = 0.5;
df = 1000;
mm_rbf = mm_rbf_fourierfeatures(size(x,2),df);

s = randi(length(x)-L+1,N,1);
Z = zeros(N,df);
% parfor_progress(N);
for i=1:N
  xhat = x(s(i):d:s(i)+L-1,:);
  xhat=xhat-repmat(mean(xhat),size(xhat,1),1); 
  xhat=xhat/max(max(abs(xhat)));
  Z(i,:) = mm_rbf(xhat,a);
% parfor_progress;
end
% parfor_progress(0);
g_idx = find(sum(isnan(Z'))==0);
Z = Z(g_idx,:);
[E,V,~] = svd(Z,'econ');
v = diag(V(1:D,1:D)).^2;
e = E(:,1:D);
phi = {};
for i=1:D
  tmp = zeros(size(x(1:d:L,:)));
  for j=1:length(g_idx);
    xhat = x(s(g_idx(j)):d:s(g_idx(j))+L-1,:);
    xhat=xhat-repmat(mean(xhat),size(xhat,1),1); xhat=xhat/max(max(abs(xhat)));
    tmp = tmp+e(j,i)*xhat;
  end
  phi{i} = tmp;
end
plot3(phi{1}(:,1),phi{1}(:,2),phi{1}(:,3),'r'); hold on;
plot3(phi{2}(:,1),phi{2}(:,2),phi{2}(:,3),'g');
plot3(phi{3}(:,1),phi{3}(:,2),phi{3}(:,3),'b'); hold off;
% xhat = x(s(1):d:s(1)+L-1,:);
% xhat=xhat-repmat(mean(xhat),size(xhat,1),1); xhat=xhat/max(max(abs(xhat)));
% hold on; plot3(xhat(:,1),xhat(:,2),xhat(:,3)); hold off;

% Steps = 1000;
% comp = zeros(Steps,D);
% shat = floor(linspace(1,length(x)-L+1,Steps));
% Zhat = Z'*e*diag(1./v);
% parfor_progress(Steps);
% parfor i=1:Steps
%   xhat = x(shat(i):d:shat(i)+L-1,:);
%   xhat=xhat-repmat(mean(xhat),size(xhat,1),1); xhat=xhat/max(max(abs(xhat)));
%   comp(i,:) = mm_rbf(xhat,a)*Zhat;
% parfor_progress;
% end
% parfor_progress(0);
% %===========================================

% plot3(comp(:,1),comp(:,2),comp(:,3))

% py = comp(:,5);
% plot(shat,tsmovavg(py,'t',10,1)); hold on; y1=min(py);y2=max(py);
% for i=1:length(ann_use)  
%    plot([ann_idx(ann_use(i,1));ann_idx(ann_use(i,1));ann_idx(ann_use(i,2));ann_idx(ann_use(i,2))],[y1;y2;y1;y2]);
% end
% hold off;

% comp2 = tsmovavg(comp,'t',10,1);
% j1=2;j2=5;
% plot3(shat,comp2(:,j1),comp2(:,j2),'Color',[0.9 0.9 0.9]); hold on;
% for i=1:length(ann_use);
% idx = shat>=ann_idx(ann_use(i,1)) & shat<=ann_idx(ann_use(i,2));
% plot3(shat(idx),comp2(idx,j1),comp2(idx,j2))
% end;
% hold off;

% %  d=20;
% %  i=randi(length(s));
% %  j=randi(length(s));
% %  xhat = x(s(i):d:s(i)+L-1,:);
% %  xhat=xhat-repmat(mean(xhat),size(xhat,1),1); xhat=xhat/max(max(abs(xhat)));
% %  xhat2 = x(s(j):d:s(j)+L-1,:);
% %  xhat2=xhat2-repmat(mean(xhat2),size(xhat2,1),1); xhat2=xhat2/max(max(abs(xhat2)));
% %  rbf(xhat,xhat2,a)
% %  sum(mm_rbf(xhat,a).*mm_rbf(xhat2,a))
