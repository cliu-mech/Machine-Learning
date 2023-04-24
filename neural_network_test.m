%% Neural Network Test
clc;clear;close all
%% Create Dataset
%rng("default");
d = 2; % dimensionality
points_per_class = 100; % number of points per class
num_classes = 3; % number of classes

n = points_per_class*num_classes;

X = zeros(n,d);
y = zeros(n,1);

for j =0:num_classes-1

    inds = points_per_class*j+1: points_per_class*(j+1);

    % Generate radius and angle for each point
    r = linspace(0.0, 1, points_per_class); % radius
    t = linspace(j*4,(j+1)*4,points_per_class) + randn(1,points_per_class)*0.2; % theta
    
    X(inds,:) = cat(1,r.*sin(t),r.*cos(t))';
    y(inds) = j+1;  % class label
end

%figure;
%scatter(X(:,1),X(:,2),[],y);
%grid minor;
%xlim([-1 1]);ylim([-1 1]);
%box on;
%% Create Test Examples
h=0.05;
x_min=min(X(:,1))-1;
x_max=max(X(:,1))+1;
y_min=min(X(:,2))-1;
y_max=max(X(:,2))+1;
xx=x_min:h:x_max;
yy=y_min:h:y_max;
[XX,YY]=meshgrid(xx,yy);

X_test=[XX(:),YY(:)];
size(X_test)

%%
nIterations=30000;
m=100;
W1=randn(d,m);
b1=zeros(1,m);
W2=randn(m,num_classes);
b2=zeros(1,num_classes);

lambda=1e-3;
%lambda=0;
eta=1e-0;
idx=1;
res=1;
loss_pre=1;
%for idx=1:nIterations
while(res>1e-4)
    [loss,gW1,gb1,gW2,gb2]=compute_loss(X,y,W1,b1,W2,b2,lambda);
    
    W1=W1-eta*gW1;
    b1=b1-eta*gb1;
    W2=W2-eta*gW2;
    b2=b2-eta*gb2;

    if(rem(idx,1000)==0)
        loss
    end

    if idx==1
        loss_pre=loss;
        idx=idx+1;
        continue;
    else
        res=abs(loss_pre-loss);
        loss_pre=loss;
    end
    idx=idx+1;
end

nDataTest=size(X_test,1);
hidden=relu(X_test*W1+ones(nDataTest,1)*b1);
scores=hidden*W2+ones(nDataTest,1)*b2;
[class,iClass]=max(scores,[],2);

%% Display model
figure;
Z=reshape(iClass,size(XX));
contourf(XX, YY, Z);
hold on;
scatter(X(:,1),X(:,2),[],y);
grid minor;
xlim([-1 1]);ylim([-1 1]);


%% inline functions
function res=relu(z)
    res=max(0,z);
end

function res=relu_prime(z)
    res=ones(size(z));
    idx=z<=0;
    res(idx)=0;
end

function [loss,gradW1,gradb1,gradW2,gradb2]=compute_loss(X,y,W1,b1,W2,b2,lambda)

    nData=size(X,1);
    % compute scores
    act=X*W1+ones(nData,1)*b1;
    hidden=relu(act);
    scores=hidden*W2+ones(nData,1)*b2; % output layer
    
    % compute probabilities
    sigma=sum(exp(scores),2);
    probabilities=exp(scores)./kron(ones(1,size(scores,2)),sigma);

    % compute cross-entropy loss
    indicator=kron(ones(1,size(scores,2)),y)==kron(ones(size(X,1),1),1:size(scores,2));
    temp=indicator.*log(probabilities);
    data_loss=mean(-sum(temp,2));

    % compute regularisation loss
    reg_loss=lambda*(norm(W1,'fro')^2+norm(W2,'fro')^2);
    
    loss=data_loss+reg_loss;
    %loss
    %pause

    % compute gradient of cross-entropy wrt class scores
    grad_entropy_loss=zeros(size(indicator));
    grad_entropy_loss(indicator==true)=probabilities(indicator==true)-1;
    grad_entropy_loss(indicator==false)=probabilities(indicator==false);
    grad_entropy_loss=grad_entropy_loss/nData;
    %grad_entropy_loss=flag.*(probabilities-1)/nData;
    %grad_entropy_loss=(probabilities-flag)/nData;
    % now backpropagate to get gradient of cross-entropy wrt parameters (W2,b2)
    gradAct=grad_entropy_loss*W2';
    gradHidden=gradAct.*relu_prime(act);
    gradW2=hidden'*grad_entropy_loss+2*lambda*W2;
    gradW1=X'*gradHidden+2*lambda*W1;
    gradb2=ones(1,nData)*grad_entropy_loss;
    gradb1=ones(1,nData)*gradHidden;
    %pause 
    
end