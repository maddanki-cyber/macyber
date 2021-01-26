%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Face - Bayesian

clear
%Loading data
data = load('data.mat');
face = data.face;
face_neutral=face(:,:,1:3:end);
face_exp=face(:,:,2:3:end);
face_illum=face(:,:,3:3:end);
[d1,d2,n] = size(face_neutral);
size(face_neutral)

d1 
d2
n
figure

%%% MAKE DATA MATRIX WHOSE ROWS CORRESPOND TO IMAGES

X1 = zeros(n,d1*d2); %data for neutral faces
X2 = zeros(n,d1*d2); %data for smiley faces
for j = 1:n   %200
    aux = face_neutral(:,:,j);
    X1(j,:) = aux(:)';
    aux = face_exp(:,:,j);
    X2(j,:) = aux(:)';
    
end

n1 = n;
n2 = n;
X = [X1;X2];
D1 = 1:n1;
D2 = n1+1:n1+n2;

%%% COMPUTE PCA

[U,Sig,V] = svd(X','econ');
nPCA = 20;
% project to nPCA-dimensional space
Y = X*U(:,1:nPCA);

figure;
hold on;
grid;
plot3(Y(D1,1), Y(D1,2), Y(D1,3),'.', 'Markersize', 20, 'color', 'k');
plot3(Y(D2,1), Y(D2,2), Y(D2,3),'.', 'Markersize', 20, 'color', 'r');
view(2);

%%% DIVIDE DATA TO TRAIN AND TEST SUBJECTS

Ntrain = 150;
Ntest = n-Ntrain;
train1 = 1:Ntrain; % neutral train indices
train2 = (n+Ntrain+1):(2*n); % smiley test indices
test1 = (Ntrain+1):n; %neutral test indices
test2 = (n+Ntrain+1):(2*n); % smiley test indices
train_idx = [train1,train2];
test_idx = [test1,test2];
Ytrain = Y(train_idx,:); % train data in PCA space
Ytest = Y(test_idx,:); % test data in PCA space




%%% Bayesian decision theory %%%
Y1=Y(D1,:);%projected data of neutral face
Y2=Y(D2,:);%projected data of expression face
mu1=mean(Y1,1);
mu2=mean(Y2,1);
fprintf('norm=%d\n', norm(mu1-mu2));

%estimate covariance matrices for each class
Y1c=Y1-ones(n1,1)*mu1; %center the data for neutral face
Y2c=Y2-ones(n2,1)*mu2; %center the data for expression face
S1=Y1c'*Y1c/n1;
S2=Y2c'*Y2c/n2;
figure;
imagesc(S1);  %neutral covariance matrix
colorbar;
figure;
imagesc(S2); %expressive covariance matrix
colorbar;

%define the descriminant functions
iS1=inv(S1);
iS2=inv(S2);
mu1=mu1';
mu2=mu2';
w0=0.5*(log(det(S2)/det(S1)))-0.5*(mu1'*iS1*mu1-mu2'*iS2*mu2);
g=@(x)-0.5*x'*(iS1-iS2)*x+x'*(iS1*mu1-iS2*mu2)+w0;

%Success rate
Y3=zeros(n,nPCA);
label=zeros(n,1);
for j=1:n
    aux=face_neutral(:,:,j);
    y=(aux(:)'*U(:,1:nPCA))'; 
    Y3(j,:)=y';
    label(j)=sign(g(y));
end
iplus=find(label>0);
iminus=find(label<0);
fprintf('#iplus=%d, #iminus=%d\n', length(iplus), length(iminus));
success_Baysian = length(iplus)/200

%%%%%%%%%%%%%%%%%%%%%%%%%%

%Face - KNN

clear
data = load('data.mat')
face = data.face
face_neutral = face(:,:,1:3:end); %for neutral faces
face_exp = face(:,:,2:3:end); % for smiley faces
face_lit = face(:,:,3,3:end); %for lit faces
[d1, d2,n] = size(face_neutral);
d1 
d2
n
figure

X1 = zeros(n, d1*d2);
X2 = zeros(n, d1*d2);
for i = 1:n%200
    aux = face_neutral(:,:,i);
    X1 (i,:) = aux(:)';
    aux1 = face_exp(:,:,i);
    X2(i,:) = aux1(:)';
end

n1 = n
n2 = n
X = [X1;X2];
D1 = 1:n1;
D2 = n1+1:n1+n2

%%PCA

[U,Sig,V] = svd(X', 'econ');
nPCA = 20;

Y = X*U(:,1:nPCA);

figure;
hold on;
grid;
plot3(Y(D1,1), Y(D1,2), Y(D1,3),'.', 'Markersize', 20, 'color', 'k');
plot3(Y(D2,1), Y(D2,2), Y(D2,3),'.', 'Markersize', 20, 'color', 'r');
view(2);
%perfect PCA



%Make test and train
Ntrain = 150;
Ntest = n-Ntrain;
train1 = 1:Ntrain; %netural train indices
train2 = (n+1) : (n+Ntrain);%smiley train indexes



test1 = (Ntrain + 1):n;%test neutral indices
test2 = (Ntrain+n+1):(2*n); %test smiley indices

train_ind = [train1, train2]; 
test_ind = [test1, test2];

Ytrain = Y(train_ind,:);%train into PCA

Ytest = Y(test_ind,:);%test into PCA



a = size(Ytrain, 1)
a

knn = 20;
labels = zeros(length(test_ind),1);
for i = 1 :length(test_ind)
    ytest = Ytest(i,:); %row vectors
    dist_sq = sum((Ytrain - ones(size(Ytrain, 1),1)*ytest).^2, 2);
    dist = sqrt(dist_sq);
    [dsort, isort] = sort(dist, 'ascend');
    %dist
    kNN = isort(1:knn);
    knn_neutral = find(kNN <= Ntrain);
    knn_smile = find(kNN > Ntrain);
    if length(knn_neutral) > length(knn_smile)
        labels(i) = 1
    else
        labels(i) = -1
    end


end

%Ytest == labels

for j= 1: (knn)
    a = kNN(j);
    a
    subplot (4,5,j)
    imagesc(face(:,:,a))
end 


acc = 0
sum = 0

for i= 1:50
    if labels(i)==1
        sum = sum +1;
    end
end
    
for i= 51:100
    if labels(i) == -1
        sum = sum+1;
    end
end

acc = sum/100

    
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Pose - KNN

%%task 2
clear
data=load('pose.mat');
pose=data.pose;
size(pose)

pose(:,:,1,1);
imagesc(pose(:,:,1,1));

pose_total = reshape(pose, [48,40,884]);

[d1, d2,n] = size(pose_total);

%figure
X1 = zeros(n, d1*d2);
X2 = zeros(n, d1*d2);
for i = 1:n%884
    aux = pose_total(:,:,i);
    X1 (i,:) = aux(:)';

end

n1 = n;
n2 = n;
X = [X1;X2];
D1 = 1:n1;
D2 = n1+1:n1+n2;




Ntrain = 10;
Ntest = 13-Ntrain;
train1 = 1:Ntrain; 


Ntrain ={};
Ntest = {};
Ntest_labels = {};
Ntrain_labels = {};

n = 10;
for i=1:68
    for j=1:13
        a = pose(:,:,j,i);
        
        %Ntrain =  [Ntrain; a]
        if j>n
            %Ntest = [Ntest; a];
            %Ntest(end+1) = a;
            %Ntest=append(Ntest,a);
            Ntest = [Ntest, a];
            Ntest_labels = [Ntest_labels, i];
        else
            %Ntrain = [Ntrain; a];
            %Ntrain(end+1) = a;
            %Ntrain=append(Ntrain,a);
            Ntrain = [Ntrain, a];
            Ntrain_labels = [Ntrain_labels, i];
        end
        
                  
    end
      
end
figure;
colormap gray;
subplot (4,5,10)


%PCA for 884 data points
ls_a = {};
%PCA to map to higher dimensional space
for i=1:204
    [U,Sig,V] = svd(Ntest{1,i}','econ');
    nPCA = 20;
    %project to nPCA-dimensional space
    Y = Ntest{1,i}*U(:,1:nPCA);
    ls_a = [ls_a, Y];
    
end

%PCA for 884 data points
ls_y = {};
%PCA to map to higher dimensional space
for i=1:680
    [U,Sig,V] = svd(Ntrain{1,i}','econ');
    nPCA = 20;
    %project to nPCA-dimensional space
    Y = Ntrain{1,i}*U(:,1:nPCA);
    ls_y = [ls_y, Y];
    end

 
test1 = 1:n;


%Y(Ntrain)
train_ind = [train1]; 
test_ind = [test1];

Ytrain = Y(train_ind,:);%train into PCA


%KNN
knn = 10;
a_test = zeros(length(ls_y));
labels = zeros(length(ls_a),1);
scores = zeros(length(ls_y));
for i = 1 :length(ls_a)
    %temp = ls_y{1}
    ytest = ls_a{1,i}; %row vectors
    %distance_matrix = (ytest - new_ls_y).^2;
    a = ones(48,20).*ytest;
    %dist_sq = sum((ls_y - ones(48,20).*ytest).^2, 2);
    %a = ones(960,1)*ytest
    for j=1:length(ls_y)
        a_test(j) = sum((ls_y{1,j}-ytest).^2, 'all');
    end
       
   %dist_sq = sum((Ytrain - ones(size(Ytrain, 1),1)*ytest).^2, 2);
    %dist = sqrt(dist_sq);
    [dsort, isort] = sort(a_test, 'ascend');
    %dist
    kNN = isort(1:knn);
    potential = zeros(length(knn));
    for k=1:knn
        potential(k) = Ntrain_labels{1,kNN};
    end
    
    knn_neutral = mode(potential);
    
    if knn_neutral == Ntest_labels{1,i}
        scores(i) = 1;
    else
        scores(i) = 0;
    end
    
    
end

score = sum(scores, 1);
accuracy = score(1)/length(scores)









