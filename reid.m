%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = '/viper_lomo.mat';

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
clear descriptors

%% set the seed of the random stream. The results reported in our CVPR 2015 paper are achieved by setting seed = 0. 
seed = 0;
rng(seed);

%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;

    nu = 1;
    beta = 1;

    n = 316;
    d = 316;
    k = d;

    p = randperm(numClass);
    
    galFea1 = galFea( p(1:numClass/2), : );
    probeFea1 = probeFea( p(1:numClass/2), : );
    
    t0 = tic;

    trainTime = toc(t0);
    
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probeFea2 = probeFea(p(numClass/2+1 : end), : );
    X1 = pca(probeFea1');
    X2 = pca(galFea1');

    X12 = pca(probeFea2');
    X22 = pca(galFea2');

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



%% Main algorithm
    for i = 1:50
        U  = (( X1 * transpose(V1)) + ( X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) * X1) + (beta* A * V2) + nu * P1 * X1);
        V2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) * X2) + (beta* transpose(A) * V1) + nu * P2 * X2);
        P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) * inv((V2 * transpose(V2)) + (La/beta)*eye(k));
    end

    D = zeros(n,n);
    for m = 1:n
        v1 = P1*X12(:,m);
        for i = 1:n
            v2 = P2*X22(:,i);
            D(m,i) = norm(((v1 - A*v2).^2));
        end
    end
    
    t0 = tic;
    dist = D;
    matchTime = toc(t0);
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    
    cms(nf,:) = EvalCMC( -dist, 1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);
title('CMC Curve');
xlabel('Rank');
ylabel('Performance');
axis([0 100 0 1])

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
