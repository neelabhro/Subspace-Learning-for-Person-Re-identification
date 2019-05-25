%CSPL Paper Replication 65% Accuracy
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'PRID450s_feature_all_GOGyMthetaHSV.mat';
%pcaFile = 'matlabPCA100.mat';

numClass = 450;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'feature_all');
galFea = feature_all(1 : numClass, :);
probeFea = feature_all(numClass + 1 : end, :);
clear feature_all

    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;

    nu = 1;
    beta = 1;

    n = 225;
    d = 100;
    k = d;

    p = randperm(numClass);
    
    galFea1 = galFea(1:numClass/2, : );
    probeFea1 = probeFea(1:numClass/2, : );

    TrainSet = zeros(450,7567);
    TrainSet(1:225 ,:) = galFea1;
    TrainSet(226: end,:) = probeFea1;
    
    t0 = tic;

    trainTime = toc(t0);
    galFea2 = galFea((numClass/2+1 : end), : );
    probeFea2 = probeFea((numClass/2+1 : end), : );
    
    TestSet = zeros(450,7567);
    TestSet(1:225 ,:) = galFea2;
    TestSet(226: end,:) = probeFea2;
    
    [X , W] = matlabPCA(TrainSet',100);
%    load(pcaFile, 'X');
%    load(pcaFile, 'W');
    
    X2 = X(:, 1:225);
    X1 = X(:, 226:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');

%    X1 = probeFea1';
%    X2 = galFea1';
    
%    X12 = probeFea2';
%    X22 = galFea2';
    TestPCA = W' * TestSet';
    X22 = TestPCA(:, 1:225);
    X12 = TestPCA(:, 226:end);

    
    %X1 = 3.* randi([0, 1], [d,n]);
    %X2 = X1 + 3;
    %X1 = X1 - 3;


    %X12 = 3.* randi([0, 1], [d,n]);
    %X22 = X12 + 3;
    %X12 = X12 - 3; 
%     X1 = 3.* randi([0, 1], [d,n]);
%     X2 = X1 + 0;
%     X1 = X1 - 0;
% 
% 
%     X12 = 3.* randi([0, 1], [d,n]);
%     X22 = X12 + 0;
%     X12 = X12 - 0;
    %X12 = pca(probeFea2');
    %X22 = pca(galFea2');

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
    for i = 1:500
        U  = (( X1 * transpose(V1)) + ( X2 * transpose(V2))) / ((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) * X1) + (beta* A * V2) + nu * P1 * X1);
        V2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) * X2) + (beta* transpose(A) * V1) + nu * P2 * X2);
        P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) * inv((V2 * transpose(V2)) + (La/beta)*eye(k));
        
    end

    
    D = zeros(n,n);
    for m = 1:n
    
        v1 = P1*(X12(:,m));
    
        for i = 1:n
            v2 = P2*(X22(:,i));
            D(m,i) = norm(((v1 - A*v2)));
        end
        
    end
    
CMC(D,100);
