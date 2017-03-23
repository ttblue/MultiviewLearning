addpath('Rmoddemeijer');
addpath(genpath('/home/mille856/Projects/Radiation/Msk_v_Unmsk_Code_Kyle/minFunc_2012'));


ann = strsplit(fileread('33_annotation.txt'),'\n');
ann_idx=[]; ann_text={};k=1;
for i=1:length(ann)
  s = ann{i};
  idx = find(isspace(s));
  if length(idx)==0, continue; end;
  ann_idx(k) = str2num(s(1:idx(1)));
  ann_text{k} = s(idx(1)+1:end);
  k=k+1;
end
ann_use = [3,8; 9,14; 19,20; 27,28; 31,36];
header = 23;
start = 1;
nrow = 5484735;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('33.csv','\t',datrows);
y = X(:,4); 
y = tsmovavg(y,'t',5,1); y = y(~isnan(y)); 
K = 500;
ami = zeros(K,1);
tau=0; fmv=inf;
parfor k=1:K
 ami(k) = information(y(1:end-k)',y(1+k:end)');
end
plot(ami)
for k=1:K
   if ami(k)<=fmv
     tau = k; fmv=ami(k);
   else
     fmv=-inf;
 end
end
tau
D = 3;
x = [];
for d=1:D
  x = [x(1:end-tau,:),y((d-1)*tau+1:end)];
end 
vecs = [x(1:end-1,:)-x(2:end,:);0,0,0];
plot(y); hold on; y1=min(y);y2=max(y);
for i=1:length(ann_use)  
   plot([ann_idx(ann_use(i,1));ann_idx(ann_use(i,1));ann_idx(ann_use(i,2));ann_idx(ann_use(i,2))],[y1;y2;y1;y2]);
end
hold off;
figure();

while 1
set(0,'CurrentFigure',1)
idx = ginput(2); idx=floor(idx(1)):floor(idx(2));
set(0,'CurrentFigure',2)
quiver3(x(idx,1),x(idx,2),x(idx,3),vecs(idx,1),vecs(idx,2),vecs(idx,3))
waitforbuttonpress;
end


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

% --Airway example
header = 22;
start = 10000;
nrow = 150000;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('31.csv','\t',datrows);
y = X(:,11); plot(y);
tau = 86;
D = 6;
x = [];
for d=1:D
  x = [x(1:end-tau,:),y((d-1)*tau+1:end)];
end
k1 = 49000;
k2 = 115000;
vecs = [x(1:end-1,1:3)-x(2:end,1:3);0,0,0];

quiver3(x(1:k1,1),x(1:k1,2),x(1:k1,3),vecs(1:k1,1),vecs(1:k1,2),vecs(1:k1,3),'b')
hold on;
quiver3(x(k1+1:k2,1),x(k1+1:k2,2),x(k1+1:k2,3),vecs(k1+1:k2,1),vecs(k1+1:k2,2),vecs(k1+1:k2,3),'g')
quiver3(x(k2+1:end,1),x(k2+1:end,2),x(k2+1:end,3),vecs(k2+1:end,1),vecs(k2+1:end,2),vecs(k2+1:end,3),'r')
hold off;


x2 = x(1:k1,:); vecs2 = vecs(1:k1,:); vecs2b=vecs2;
idx = knnsearch(x2,x2,'K',150);
for i=1:length(x2)
  vecs2b(i,:) = mean(vecs(idx(i,:)',:),1);
end
quiver3(x2(:,1),x2(:,2),x2(:,3),vecs2b(:,1),vecs2b(:,2),vecs2b(:,3),'b')

[wt,period] = cwt(x2(:,1),1/250);
yrec = icwt(wt,period,[min(period) 0.00005],'SignalMean',mean(x2(:,1)))';
xrec = [];
for d=1:D
  xrec = [xrec(1:end-tau,:),yrec((d-1)*tau+1:end)];
end
vecsrec = [xrec(1:end-1,1:3)-xrec(2:end,1:3);0,0,0];
hold on;
quiver3(xrec(:,1),xrec(:,2),xrec(:,3),vecsrec(:,1),vecsrec(:,2),vecsrec(:,3),'g')
hold off;
plot3(xrec(:,1),xrec(:,2),xrec(:,3))

K = 5000;
skip = 100;
Period = zeros(floor(k1/skip),1);
parfor i=1:floor(k1/skip)
idx = i*skip:i*skip+K;
x2 = x(idx,:);
t = X(idx,1);
Nneighbors = 10;
[idx,dist] = knnsearch(x2,x2,'K',Nneighbors);
p = t(idx)-repmat(t(idx(:,1)),1,Nneighbors);
p = reshape(p(:,2:end),numel(p(:,2:end)),1); p = abs(p(p~=0));
mincost = inf; period=-1;
tmp = []; j=1;
for P=linspace(min(p)+0.1,max(p),500)
  cost = mod(p,P)/P;
  cost = sum(cost.*(1-cost));
  tmp(j) = cost; j=j+1;
  if cost<mincost,
    mincost = cost;
    period = P;
  end
end
Period(i)=period;
end

x3 = x(1:k1,1:3)-repmat(mean(x(1:k1,1:3),1),length(1:k1),1); 
%x3 = x(k1+1:k2,1:3)-repmat(mean(x(k1+1:k2,1:3),1),length(k1+1:k2),1); 
tmp = pca(x3); t = atan2(x3*tmp(:,2),x3*tmp(:,1));
%R = 5; knots = (rand(R*R+2*R,size(x3,2))-0.5)*max(max(abs(x3),[],1))*1;
J = 2*1+1; knots = (rand(J,size(x3,2))-0.5)*max(max(abs(x3),[],1))*1;
options = [];
options.Method = 'lbfgs';
options.display = 'none';
dt = linspace(1/1500,1/1000,500);
cost = zeros(length(dt),1);
parfor i=1:length(dt)
  xout = minFunc(@(x) loop3(x,x3,2*pi*dt(i)),reshape(knots,numel(knots),1),options);
  cost(i) = loop3(xout,x3,2*pi*dt(i));
end
[c,i] = min(cost)
J = 2*15+1; knots = (rand(J,size(x3,2))-0.5)*max(max(abs(x3),[],1))*1;
xout = minFunc(@(x) loop3(x,x3,2*pi*dt(i)),reshape(knots,numel(knots),1),options);
knots=reshape(xout(1:numel(knots)),size(knots,1),size(x3,2));% t = xout(numel(knots)+1:end);
%  vecs = [x3(1:end-1,1:3)-x3(2:end,1:3);0,0,0];
%  quiver3(x3(:,1),x3(:,2),x3(:,3),vecs(:,1),vecs(:,2),vecs(:,3),'b')
%  hold on; plot3(knots(:,1),knots(:,2),knots(:,3)); hold off;
%knots = fmincon(@(c) loop(c,x3,t),knots);

t2 = linspace(-pi,pi,1000)';
J = size(knots,1);
n = (J-1)/2;
coeffn = 2.^n./(n+1)./nchoosek(J,n); % const
B = coeffn*((1+cos(t2(:,ones(1,J))+repmat(2*pi*(0:J-1)/J,length(t2),1))).^n)*knots;
%  B = [];
%     for n=1:R
%      J = 2*n+1;
%      coeffn = 2.^n./(n+1)./nchoosek(J,n); % const
%      B_n = coeffn*(1+cos(t2(:,ones(1,J))+repmat(2*pi*(0:J-1)/J,length(t2),1))).^n; % N by J
%      B = [B,B_n];
%     end
%  B = B*knots;
vecs = [x3(1:end-1,1:3)-x3(2:end,1:3);0,0,0];
quiver3(x3(:,1),x3(:,2),x3(:,3),vecs(:,1),vecs(:,2),vecs(:,3),'b')
hold on; plot3(B(:,1),B(:,2),B(:,3)); hold off;

vecs = [x3(1:end-1,1:2)-x3(2:end,1:2);0,0];
quiver(x3(:,1),x3(:,2),vecs(:,1),vecs(:,2),'b')
hold on; plot(B(:,1),B(:,2)); hold off;

tmp = x3-coeffn*((1+cos(t(:,ones(1,J))+repmat(2*pi*(0:J-1)/J,length(t),1))).^n)*knots;
plot3(tmp(:,1),tmp(:,2),tmp(:,3));
%--

%====================================================
%====================================================
%====================================================
header = 23;
start = 1;
nrow = 150000;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('33.csv','\t',datrows);
y = X(:,4); ytmp=y(1:10000);

%  G = @(t,x) [x(2)+x(1)*x(2)-x(1).^3;-x(1)+x(2).^2];
%  
%  Y = zeros(500,20);
%  X = {};
%  for t=1:20
%  [timevals,ytmp] = ode45(G,[0 100],rand(2,1));
%  ytmp = ytmp(1:501,:);
%  X{t} = FeatureFnc(ytmp(1:end-1,1),ytmp(1:end-1,2),rand(size(ytmp,1)-1,1));
%  mn = mean(X{t},1); mn(1)=0; sd = std(X{t},1); sd(1)=1;
%  X{t} = (X{t}-repmat(mn,size(X{t},1),1))*spdiags(1./sd',0,size(X{t},2),size(X{t},2));
%  Y(:,t)=ytmp(2:end,1)-ytmp(1:end-1,1);
%  end
%  W = L21_block_regression(Y,X,3,1e-6,500,0);
%  idx = mean(abs(W),2)>1e-3; names(idx)
%  
%  y=[];
%  for i=1:30
%  [timevals,ytmp] = ode45(G,[0 100],rand(2,1));
%  y = [y;ytmp(1:500,1)];
%  end;
%  ytmp = y(1:500);

K = 500;
ami = zeros(K,1);
tau=0; fmv=inf;
for k=1:K
 ami(k) = information(ytmp(1:end-k)',ytmp(1+k:end)');
end
plot(ami)
for k=1:K
   if ami(k)<=fmv
     tau = k; fmv=ami(k);
   else
     fmv=-inf;
 end
end
tau
%---------
Rsqrd = @(Ytest,Xtest,Y,X,t,idx) 1-sum((Ytest(:,t)-Xtest{t}(:,idx)*((X{t}(:,idx)'*X{t}(:,idx))\(X{t}(:,idx)'*Y(:,t)))).^2)/sum((Ytest(:,t)-mean(Y(:,t))).^2);
FeatureFnc = @(x1,x2,x3) [ones(size(x1,1),1),x1,x2,x3,x1.*x2,x1.*x3,x2.*x3,x1.^2,x2.^2,x3.^2, ...
          x1.^2.*x2,x1.^2.*x3,x1.*x2.*x3,x1.^3,x1.*x2.^2,x1.*x3.^2,x2.^2.*x3,x2.^3,x2.*x3.^2,x3.^3, ...
          x1.^3.*x2,x1.^3.*x3,x1.^2.*x2.*x3,x1.^4,x1.^2.*x2.^2,x1.^2.*x3.^2,x1.*x2.^2.*x3,x1.*x2.^3,x1.*x2.*x3.^2,x1.*x3.^3, ...
          x2.^3.*x3,x2.^4,x2.^2.*x3.^2,x2.*x3.^3,x3.^4];
names = {'1','x','y','z','x*y','x*z','y*z','x^2','y^2','z^2','x^2*y','x^2*z','x*y*z','x^3','x*y^2','x*z^2','y^2*z','y^3','y*z^2','z^3','x^3*y','x^3*z','x^2*y*z','x^4','x^2*y^2','x^2*z^2','x*y^2*z','x*y^3','x*y*z^2','x*z^3','y^3*z','y^4','y^2*z^2','y*z^3','z^4'};
T = 10; T = floor(linspace(1,length(y),T));
K = 5000;
Y1 = zeros(K,length(T)-1); Y2 = zeros(K,length(T)-1); Y3 = zeros(K,length(T)-1);
Ytest1 = zeros(K,length(T)-1); Ytest2 = zeros(K,length(T)-1); Ytest3 = zeros(K,length(T)-1);
X = {}; Xtest = {};
for t=1:length(T)-1
  x1 = y(T(t):T(t)+K-1);
  x2 = y(T(t)+tau:T(t)+K+tau-1);
  x3 = y(T(t)+2*tau:T(t)+K+2*tau-1);
  X{t} = FeatureFnc(x1,x2,x3);
  Y1(:,t) = y(T(t)+1:T(t)+K)-x1;
  Y2(:,t) = y(T(t)+tau+1:T(t)+K+tau)-x2;
  Y3(:,t) = y(T(t)+2*tau+1:T(t)+K+2*tau)-x3;
  mn = mean(X{t},1); mn(1)=0; sd = std(X{t},1); sd(1)=1;
  X{t} = (X{t}-repmat(mn,K,1))*spdiags(1./sd',0,size(X{t},2),size(X{t},2));
  %---
  x1 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1);
  x2 = y(T(t)+K+3*tau:T(t)+2*K+3*tau-1);
  x3 = y(T(t)+K+4*tau:T(t)+2*K+4*tau-1);
  Xtest{t} = FeatureFnc(x1,x2,x3);
  Ytest1(:,t) = y(T(t)+K+2*tau+1:T(t)+2*K+2*tau)-x1;
  Ytest2(:,t) = y(T(t)+K+3*tau+1:T(t)+2*K+3*tau)-x2;
  Ytest3(:,t) = y(T(t)+K+4*tau+1:T(t)+2*K+4*tau)-x3;
  Xtest{t} = (Xtest{t}-repmat(mn,K,1))*spdiags(1./sd',0,size(X{t},2),size(X{t},2));
  %---
end
for t=1:length(T)-1
  idx = Y1(:,t)>quantile(reshape(Y1,numel(Y1),1),0.99) | Y2(:,t)>quantile(reshape(Y2,numel(Y2),1),0.99) | Y3(:,t)>quantile(reshape(Y3,numel(Y3),1),0.99);
  Y1(idx,t) = 0; Y2(idx,t) = 0; Y3(idx,t) = 0; X{t}(idx,:) = 0;
  idx = Ytest1(:,t)>quantile(reshape(Ytest1,numel(Ytest1),1),0.99) | Ytest2(:,t)>quantile(reshape(Ytest2,numel(Ytest2),1),0.99) | Ytest3(:,t)>quantile(reshape(Ytest3,numel(Ytest3),1),0.99);
  Ytest1(idx,t) = 0; Ytest2(idx,t) = 0; Ytest3(idx,t) = 0; Xtest{t}(idx,:) = 0;
end
scatter3(X{1}(:,2),X{1}(:,3),X{1}(:,4))
rsq = zeros(15,3); cnt = zeros(15,3);
for L=1:15
  W = L21_block_regression(Y1,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest1,Xtest,Y1,X,t,idx)/(length(T)-1); end;
  rsq(L,1) = R; cnt(L,1) = sum(idx);
  W = L21_block_regression(Y2,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest2,Xtest,Y2,X,t,idx)/(length(T)-1); end;
  rsq(L,2) = R; cnt(L,2) = sum(idx);
  W = L21_block_regression(Y3,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest3,Xtest,Y3,X,t,idx)/(length(T)-1); end;
  rsq(L,3) = R; cnt(L,3) = sum(idx);
end
%sortrows([rsq,cnt,(1:15)'])
[rsq,cnt,(1:15)']
l = 13*5;
W = L21_block_regression(Y1,X,l,1e-6,500,0);
idx1 = mean(abs(W),2)>1e-2;
names(idx1)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest1,Xtest,Y1,X,t,idx1)/(length(T)-1); end; R
W = L21_block_regression(Y2,X,l,1e-6,500,0);
idx2 = mean(abs(W),2)>1e-2;
names(idx2)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest2,Xtest,Y2,X,t,idx2)/(length(T)-1); end; R
W = L21_block_regression(Y3,X,l,1e-6,500,0);
idx3 = mean(abs(W),2)>1e-2;
names(idx3)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest3,Xtest,Y3,X,t,idx3)/(length(T)-1); end; R

figure(1); %close all;
t=6; w = 2*250; 
L2 = K*0.01;
x1 = y(T(t):T(t)+K-1);
x2 = y(T(t)+tau:T(t)+K+tau-1);
x3 = y(T(t)+2*tau:T(t)+K+2*tau-1);
xtrain = FeatureFnc(x1,x2,x3);
mn = mean(xtrain,1); mn(1)=0; sd = std(xtrain,1); sd(1)=1;
ytrain1 = y(T(t)+1:T(t)+K)-x1;
ytrain2 = y(T(t)+tau+1:T(t)+K+tau)-x2;
ytrain3 = y(T(t)+2*tau+1:T(t)+K+2*tau)-x3;
xtrain = (xtrain-repmat(mn,K,1))*spdiags(1./sd',0,size(xtrain,2),size(xtrain,2));
x1 = y(T(t)+K:T(t)+2*K-1); %x1 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1);
x2 = y(T(t)+K+tau:T(t)+2*K+tau-1); %x2 = y(T(t)+K+3*tau:T(t)+2*K+3*tau-1);
x3 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1); %x3 = y(T(t)+K+4*tau:T(t)+2*K+4*tau-1);
xtest = FeatureFnc(x1,x2,x3);
xtest = (xtest-repmat(mn,K,1))*spdiags(1./sd',0,size(xtest,2),size(xtest,2));
ytest3 = y(T(t)+K+2*tau+1:T(t)+2*K+2*tau); %ytest = y(T(t)+K+4*tau+1:T(t)+2*K+4*tau);
beta1 = (xtrain(:,idx1)'*xtrain(:,idx1)+L2*eye(sum(idx1)) )\(xtrain(:,idx1)'*ytrain1);
beta2 = (xtrain(:,idx2)'*xtrain(:,idx2)+L2*eye(sum(idx2)) )\(xtrain(:,idx2)'*ytrain2);
beta3 = (xtrain(:,idx3)'*xtrain(:,idx3)+L2*eye(sum(idx3)) )\(xtrain(:,idx3)'*ytrain3);
plot(ytest3);
hold on; plot(x3+xtest(:,idx3)*beta3); hold off;
xlim([1,w]);
lims = ylim();
figure(2);

ypred = zeros(K,3);
ypred(1,:) = [y(T(t)+K+1),y(T(t)+K+tau+1),y(T(t)+K+2*tau+1)];
ypredb = [y(T(t):T(t)+K+2*tau+1);zeros(K,1)];
subset = @(y,idx) y(idx);
FeatureFnc2 = @(x,mn,sd) (FeatureFnc(x(1),x(2),x(3))-mn)*spdiags(1./sd',0,size(names,2),size(names,2));
govfnc = @(x,mn,sd) [subset(FeatureFnc2(x,mn,sd),idx1)*beta1;subset(FeatureFnc2(x,mn,sd),idx2)*beta2;subset(FeatureFnc2(x,mn,sd),idx3)*beta3];
govfnc2 = @(t,x) govfnc(x,mn,sd);
%[timevals,ypred45] = ode45(govfnc2,[0 w],ypred(1,:)');
for i=2:K
  ypred(i,:) = ypred(i-1,:)+govfnc(ypred(i-1,:),mn,sd)';

  x1 = ypredb(K+i);
  x2 = ypredb(K+i+tau);
  x3 = ypredb(K+i+2*tau);
  xtest = FeatureFnc(x1,x2,x3);
  xtest = (xtest-mn)*spdiags(1./sd',0,size(xtest,2),size(xtest,2));
  ypredb(K+i+2*tau+1) = x3+xtest(:,idx3)*beta3;
end
plot(ytest3(1:w));
hold on; plot(ypred(1:w,3)); 
plot(ypredb(K+2*tau+2:K+2*tau+2+w)); 
%plot(timevals,ypred45(:,3));
hold off;
ylim(lims);


%====================================================
%====================================================
%====================================================
header = 23;
start = 1;
nrow = 150000;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('33.csv','\t',datrows);
y = X(:,4); ytmp=y(1:10000);
K = 500;
ami = zeros(K,1);
tau=0; fmv=inf;
for k=1:K
 ami(k) = information(ytmp(1:end-k)',ytmp(1+k:end)');
end
plot(ami)
for k=1:K
   if ami(k)<=fmv
     tau = k; fmv=ami(k);
   else
     fmv=-inf;
 end
end
tau
%---------
Rsqrd = @(Ytest,Xtest,Y,X,t,idx) 1-sum((Ytest(:,t)-Xtest{t}(:,idx)*((X{t}(:,idx)'*X{t}(:,idx))\(X{t}(:,idx)'*Y(:,t)))).^2)/sum((Ytest(:,t)-mean(Y(:,t))).^2);
FeatureFnc = @(x1,x2,x3) [ones(size(x1,1),1),x1,x2,x3,x1.*x2,x1.*x3,x2.*x3,x1.^2,x2.^2,x3.^2, ...
          x1.^2.*x2,x1.^2.*x3,x1.*x2.*x3,x1.^3,x1.*x2.^2,x1.*x3.^2,x2.^2.*x3,x2.^3,x2.*x3.^2,x3.^3, ...
          x1.^3.*x2,x1.^3.*x3,x1.^2.*x2.*x3,x1.^4,x1.^2.*x2.^2,x1.^2.*x3.^2,x1.*x2.^2.*x3,x1.*x2.^3,x1.*x2.*x3.^2,x1.*x3.^3, ...
          x2.^3.*x3,x2.^4,x2.^2.*x3.^2,x2.*x3.^3,x3.^4];
T = 10; T = floor(linspace(1,length(y),T));
K = 5000;
;
for t=1:length(T)-1
  x1 = y(T(t):T(t)+K-1);
  x2 = y(T(t)+tau:T(t)+K+tau-1);
  x3 = y(T(t)+2*tau:T(t)+K+2*tau-1);
  X{t} = FeatureFnc(x1,x2,x3);
  Y1(:,t) = y(T(t)+1:T(t)+K)-x1;
  Y2(:,t) = y(T(t)+tau+1:T(t)+K+tau)-x2;
  Y3(:,t) = y(T(t)+2*tau+1:T(t)+K+2*tau)-x3;
  mn = mean(X{t},1); mn(1)=0; sd = std(X{t},1); sd(1)=1;
  X{t} = (X{t}-repmat(mn,K,1))*spdiags(1./sd',0,size(X{t},2),size(X{t},2));
  %---
  x1 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1);
  x2 = y(T(t)+K+3*tau:T(t)+2*K+3*tau-1);
  x3 = y(T(t)+K+4*tau:T(t)+2*K+4*tau-1);
  Xtest{t} = FeatureFnc(x1,x2,x3);
  Ytest1(:,t) = y(T(t)+K+2*tau+1:T(t)+2*K+2*tau)-x1;
  Ytest2(:,t) = y(T(t)+K+3*tau+1:T(t)+2*K+3*tau)-x2;
  Ytest3(:,t) = y(T(t)+K+4*tau+1:T(t)+2*K+4*tau)-x3;
  Xtest{t} = (Xtest{t}-repmat(mn,K,1))*spdiags(1./sd',0,size(X{t},2),size(X{t},2));
  %---
end
for t=1:length(T)-1
  idx = Y1(:,t)>quantile(reshape(Y1,numel(Y1),1),0.99) | Y2(:,t)>quantile(reshape(Y2,numel(Y2),1),0.99) | Y3(:,t)>quantile(reshape(Y3,numel(Y3),1),0.99);
  Y1(idx,t) = 0; Y2(idx,t) = 0; Y3(idx,t) = 0; X{t}(idx,:) = 0;
  idx = Ytest1(:,t)>quantile(reshape(Ytest1,numel(Ytest1),1),0.99) | Ytest2(:,t)>quantile(reshape(Ytest2,numel(Ytest2),1),0.99) | Ytest3(:,t)>quantile(reshape(Ytest3,numel(Ytest3),1),0.99);
  Ytest1(idx,t) = 0; Ytest2(idx,t) = 0; Ytest3(idx,t) = 0; Xtest{t}(idx,:) = 0;
end
scatter3(X{1}(:,2),X{1}(:,3),X{1}(:,4))
rsq = zeros(15,3); cnt = zeros(15,3);
for L=1:15
  W = L21_block_regression(Y1,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest1,Xtest,Y1,X,t,idx)/(length(T)-1); end;
  rsq(L,1) = R; cnt(L,1) = sum(idx);
  W = L21_block_regression(Y2,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest2,Xtest,Y2,X,t,idx)/(length(T)-1); end;
  rsq(L,2) = R; cnt(L,2) = sum(idx);
  W = L21_block_regression(Y3,X,L*5,1e-6,500,0); idx = mean(abs(W),2)>1e-2;
  R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest3,Xtest,Y3,X,t,idx)/(length(T)-1); end;
  rsq(L,3) = R; cnt(L,3) = sum(idx);
end
%sortrows([rsq,cnt,(1:15)'])
[rsq,cnt,(1:15)']
l = 13*5;
W = L21_block_regression(Y1,X,l,1e-6,500,0);
idx1 = mean(abs(W),2)>1e-2;
names(idx1)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest1,Xtest,Y1,X,t,idx1)/(length(T)-1); end; R
W = L21_block_regression(Y2,X,l,1e-6,500,0);
idx2 = mean(abs(W),2)>1e-2;
names(idx2)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest2,Xtest,Y2,X,t,idx2)/(length(T)-1); end; R
W = L21_block_regression(Y3,X,l,1e-6,500,0);
idx3 = mean(abs(W),2)>1e-2;
names(idx3)
R = 0; for t=1:length(T)-1, R = R + Rsqrd(Ytest3,Xtest,Y3,X,t,idx3)/(length(T)-1); end; R

figure(1); %close all;
t=6; w = 2*250; 
L2 = K*0.01;
x1 = y(T(t):T(t)+K-1);
x2 = y(T(t)+tau:T(t)+K+tau-1);
x3 = y(T(t)+2*tau:T(t)+K+2*tau-1);
xtrain = FeatureFnc(x1,x2,x3);
mn = mean(xtrain,1); mn(1)=0; sd = std(xtrain,1); sd(1)=1;
ytrain1 = y(T(t)+1:T(t)+K)-x1;
ytrain2 = y(T(t)+tau+1:T(t)+K+tau)-x2;
ytrain3 = y(T(t)+2*tau+1:T(t)+K+2*tau)-x3;
xtrain = (xtrain-repmat(mn,K,1))*spdiags(1./sd',0,size(xtrain,2),size(xtrain,2));
x1 = y(T(t)+K:T(t)+2*K-1); %x1 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1);
x2 = y(T(t)+K+tau:T(t)+2*K+tau-1); %x2 = y(T(t)+K+3*tau:T(t)+2*K+3*tau-1);
x3 = y(T(t)+K+2*tau:T(t)+2*K+2*tau-1); %x3 = y(T(t)+K+4*tau:T(t)+2*K+4*tau-1);
xtest = FeatureFnc(x1,x2,x3);
xtest = (xtest-repmat(mn,K,1))*spdiags(1./sd',0,size(xtest,2),size(xtest,2));
ytest3 = y(T(t)+K+2*tau+1:T(t)+2*K+2*tau); %ytest = y(T(t)+K+4*tau+1:T(t)+2*K+4*tau);
beta1 = (xtrain(:,idx1)'*xtrain(:,idx1)+L2*eye(sum(idx1)) )\(xtrain(:,idx1)'*ytrain1);
beta2 = (xtrain(:,idx2)'*xtrain(:,idx2)+L2*eye(sum(idx2)) )\(xtrain(:,idx2)'*ytrain2);
beta3 = (xtrain(:,idx3)'*xtrain(:,idx3)+L2*eye(sum(idx3)) )\(xtrain(:,idx3)'*ytrain3);
plot(ytest3);
hold on; plot(x3+xtest(:,idx3)*beta3); hold off;
xlim([1,w]);
lims = ylim();
figure(2);

ypred = zeros(K,3);
ypred(1,:) = [y(T(t)+K+1),y(T(t)+K+tau+1),y(T(t)+K+2*tau+1)];
ypredb = [y(T(t):T(t)+K+2*tau+1);zeros(K,1)];
subset = @(y,idx) y(idx);
FeatureFnc2 = @(x,mn,sd) (FeatureFnc(x(1),x(2),x(3))-mn)*spdiags(1./sd',0,size(names,2),size(names,2));
govfnc = @(x,mn,sd) [subset(FeatureFnc2(x,mn,sd),idx1)*beta1;subset(FeatureFnc2(x,mn,sd),idx2)*beta2;subset(FeatureFnc2(x,mn,sd),idx3)*beta3];
govfnc2 = @(t,x) govfnc(x,mn,sd);
%[timevals,ypred45] = ode45(govfnc2,[0 w],ypred(1,:)');
for i=2:K
  ypred(i,:) = ypred(i-1,:)+govfnc(ypred(i-1,:),mn,sd)';

  x1 = ypredb(K+i);
  x2 = ypredb(K+i+tau);
  x3 = ypredb(K+i+2*tau);
  xtest = FeatureFnc(x1,x2,x3);
  xtest = (xtest-mn)*spdiags(1./sd',0,size(xtest,2),size(xtest,2));
  ypredb(K+i+2*tau+1) = x3+xtest(:,idx3)*beta3;
end
plot(ytest3(1:w));
hold on; plot(ypred(1:w,3)); 
plot(ypredb(K+2*tau+2:K+2*tau+2+w)); 
%plot(timevals,ypred45(:,3));
hold off;
ylim(lims);

%====================================================

header = 22;
start = 10000;
nrow = 150000;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('31.csv','\t',datrows);
y = X(:,11); plot(y);
tau = 86;
D = 6;
x = [];
for d=1:D
  x = [x(1:end-tau,:),y((d-1)*tau+1:end)];
end
k1 = 49000;
k2 = 115000;
x1 = x(1:k1,:); x1=x1-repmat(mean(x1),size(x1,1),1); x1=x1/max(max(abs(x1)));
x2 = x(k1+1:k2,:);  x2=x2-repmat(mean(x2),size(x2,1),1); x2=x2/max(max(abs(x2)));
x3 = x(k2+1:end,:);  x3=x3-repmat(mean(x3),size(x3,1),1); x3=x3/max(max(abs(x3)));


%  FeatureFnc = @(x1,x2,x3) [ones(size(x1,1),1),exp(x1),exp(x2),exp(x3),exp(-x1),exp(-x2),exp(-x3),sin(x1),sin(x2),sin(x3), ...
%            x1,x2,x3,x1.*x2,x1.*x3,x2.*x3,x1.^2,x2.^2,x3.^2, ...
%            x1.^2.*x2,x1.^2.*x3,x1.*x2.*x3,x1.^3,x1.*x2.^2,x1.*x3.^2,x2.^2.*x3,x2.^3,x2.*x3.^2,x3.^3, ...
%            x1.^3.*x2,x1.^3.*x3,x1.^2.*x2.*x3,x1.^4,x1.^2.*x2.^2,x1.^2.*x3.^2,x1.*x2.^2.*x3,x1.*x2.^3,x1.*x2.*x3.^2,x1.*x3.^3, ...
%            x2.^3.*x3,x2.^4,x2.^2.*x3.^2,x2.*x3.^3,x3.^4];
%  Fnc = @(x) FeatureFnc(x(:,1),x(:,2),x(:,3));

split1 = floor(length(x1)*0.8);
split2 = floor(length(x2)*0.8);
split3 = floor(length(x3)*0.8);
T=20;
fit1 = DaD(x1(1:split1,:),x1(1+split1:end,:),@Feature_Fnc,T,100,0.0000001,500);

T=5000;
xhat = zeros(T,D);
%idx = 1+split1+randi(length(x1)-T-1-split1);
idx = randi(length(x1)-T-1);
xhat(1,:) = x1(idx,:);
for t=2:T
  xhat(t,:) = Feature_Fnc(xhat(t-1,:))*fit1;
end
plot(x1(idx:idx+T,D)); hold on; plot(xhat(:,D)); hold off;

%====================================================

header = 22;
start = 10000;
nrow = 150000;
datrows = [header+start-1 0 header+start+nrow-2 11];
X = dlmread('31.csv','\t',datrows);
y = X(:,11); plot(y);
tau = 86;
D = 3;
x = [];
for d=1:D
  x = [x(1:end-tau,:),y((d-1)*tau+1:end)];
end
k1 = 49000;
k2 = 115000;
x1 = x(1:k1,:); x1=x1-repmat(mean(x1),size(x1,1),1); x1=x1/max(max(abs(x1)));
x2 = x(k1+1:k2,:);  x2=x2-repmat(mean(x2),size(x2,1),1); x2=x2/max(max(abs(x2)));
x3 = x(k2+1:end,:);  x3=x3-repmat(mean(x3),size(x3,1),1); x3=x3/max(max(abs(x3)));

M = [convdot(x1,x1), convdot(x1,x2), convdot(x1,x3); convdot(x2,x1), convdot(x2,x2), convdot(x2,x3); convdot(x3,x1), convdot(x3,x2), convdot(x3,x3)];
[u,v] = eig(M)
