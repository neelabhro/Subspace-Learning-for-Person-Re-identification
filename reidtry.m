%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

%x1 = imread('VIPeR/cam_a/000_45.bmp');
%imshow(x1);
%figure;
%x2 = imread('VIPeR/cam_b/000_45.bmp');  
%imshow(x2);
%% load the extracted LOMO features
numClass = 632;
numFolds = 10;
numRanks = 100;

%For loading the extracted LOMO features
load('/viper_lomo.mat', 'descriptors');
%Gallery Features
%galFea = descriptors(1 : numClass, 1:100);
%galFea = descriptors(1 : numClass,:);
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
%galFea = transpose(galFea);
%Probe Features
%probeFea = descriptors(numClass*2 + 1 : numClass*3, :);
%probeFea = transpose(probeFea);
%probFea = descriptors(numClass + 1 : end, 1:100);
%clear descriptors
galFea1 = galFea(1:numClass/2, : );
probFea1 = probeFea(1:numClass/2, : );

galFea2 = galFea(numClass/2+1 : end, : );
probFea2 = probeFea(numClass/2+1 : end, : );

%X1 = images;

Lu = 0.05;
Lv = 0.2;
La = 0.2;
Lp = 0.2;

nu = 1;
beta = 1;

n = 316;
d = 100;
k = d;

%probeFeaT = transpose(probeFea);
probe_PCA1 = matlabPCA(probFea1',100);
X12 = matlabPCA(probFea2',100);
%for m=1:216
%    probe_PCA = pca(transpose(probe_PCA));
%end
%probe_PCA = transpose(probe_PCA);

%x1 = zeros(d,n);
%for m = 1:n
%    x1(:,m) = probe_PCA(:,m);
%end  

gal_PCA2 = matlabPCA(galFea1',100);
X22 = matlabPCA(galFea2',100);
%for m=1:216
%    gal_PCA = pca(transpose(gal_PCA));
%end
%gal_PCA = transpose(gal_PCA);
%d = length(probe_PCA);
%X1 = transpose(probFea);
%X2 = transpose(galFea);
X1 = probe_PCA1;
X2 = gal_PCA2;

% n is the size of the sample set
% d is the feature dimension equal to 100
% Components of Probe set X1 and gallery set X2 are matched image descriptors
% K is the number of latent factors and is equal to d
% U is the basis matrix capturing latent intrinsic structure of the input
% data matrix
% V1 and V2 (kxn) indicate the semantic representation of X1 and X2

% X(dxn) = U(dxk)*V(kxn)
%X1 = randi([0, 1], [d,n]);
%X2 = randi([0, 1], [d,n]);

U  = randi([0, 1], [d,k]);
V1 = randi([0, 1], [k,n]);
V2 = randi([0, 1], [k,n]);
A  = randi([0, 1], [k,k]);
P1 = randi([0, 1], [k,d]);
P2 = randi([0, 1], [k,d]);

%V1T = V1';

%% Main algorithm
U  = (( X1 * transpose(V1)) + ( X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
V1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) * X1) + (beta* A * V2) + nu * P1 * X1);
V2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) * X2) + (beta* transpose(A) * V1) + nu * P2 * X2);
P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
%P1 = P1 ./ max(norm(P1));
P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
A  = (V1 * transpose(V2)) * inv((V2 * transpose(V2)) + (La/beta)*eye(k));

D = zeros(n,n);
%m = 1:1:632;
%x1 = X1(:,m);
%for m = 1:n
%    v1 = P1*X1(:,m);
%    v2 = P2*X2(m,:);
%    D(m) = norm((v1 - A*v2).^2);
%end
for m = 1:n
    v1 = P1*X12(:,m);
    for i = 1:n
        v2 = P2*X22(:,i);
        D(m,i) = norm(((v1 - A*v2).^2));
    end
end
%D = D(:,1);

clear descriptors

%% set the seed of the random stream. The results reported in our CVPR 2015 paper are achieved by setting seed = 0. 
seed = 0;
rng(seed);

%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
    p = randperm(numClass);
    
    galFea1 = galFea( p(1:numClass/2), : );
    probFea1 = probeFea( p(1:numClass/2), : );
    
    t0 = tic;
%    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');

    %{
    %% if you need to set different parameters other than the defaults, set them accordingly
    options.lambda = 0.001;
    options.qdaDims = -1;
    options.verbose = true;
    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)', options);
    %}
    
    clear galFea1 probFea1
    trainTime = toc(t0);
    
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probFea2 = probeFea(p(numClass/2+1 : end), : );
    
    t0 = tic;
%    dist = MahDist(M, galFea2 * W, probFea2 * W);
    dist = D;
    clear galFea2 probFea2 M W
    matchTime = toc(t0);
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    
    cms(nf,:) = EvalCMC( -dist, 1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms)*100;
plot(1 : numRanks, meanCms);
title('CMC Curve');
xlabel('Rank');
ylabel('Performance');
axis([0 100 0 100])

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
