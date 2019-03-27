% Same W learned for Probe and Gallery set, distance on test set
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'viper_lomo.mat';
pcaFile = 'matlabPCA316.mat';

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
clear descriptors

    L = 0.00001;
    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;
    Lw = 0.01;

    nu =0;
    beta = 1;

    n = 316;
    d = 316;
    k = d;

    p = randperm(numClass);
    
    galFea1 = galFea(p(1:numClass/2), : );
    probeFea1 = probeFea(p(1:numClass/2), : );

    TrainSet = zeros(632,26960);
    TrainSet(1:316 ,:) = galFea1;
    TrainSet(317: end,:) = probeFea1;
    
    t0 = tic;

    trainTime = toc(t0);
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probeFea2 = probeFea(p(numClass/2+1 : end), : );
    
    TestSet = zeros(632,26960);
    TestSet(1:316 ,:) = galFea2;
    TestSet(317: end,:) = probeFea2;
    
    %[X , W] = matlabPCA(TrainSet',100);
    load(pcaFile, 'X');
    load(pcaFile, 'W');
    
    X2 = X(:, 1:316);
    X1 = X(:, 317:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');
    
    TestPCA = W' * TestSet';
    X22 = TestPCA(:, 1:316);
    X12 = TestPCA(:, 317:end);

    
    

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
for i = 1:10
    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P2 = randi([0, 1], [k,d]);
    W2 = randi([0, 1], [d,k]);
    %V12 = randi([0, 1], [k,n]);
    %V22 = randi([0, 1], [k,n]);


    
%% Main algorithm
    for i = 1:500
        U  = (( W2*X1 * transpose(V1)) + ( W2*X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2* X1) + (beta* A * V2) + nu * P1 * W2*X1);
        V2 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2* X2) + (beta* transpose(A) * V1) + nu * P2*W2 * X2);
        %P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        %P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) /((V2 * transpose(V2)) + (La/beta)*eye(k));
        W2 = (X1*V1'*U' + X2*V2'*U')/((X1*X1' + X2*X2' - Lw*eye(k)));
        
        for j = 1:k
            W2(:,j) = W2(:,j)./norm(W2(:,j));
        end
        

        %W2 = W1';
        %for j = 1:50
        %    Vw = 2*W'*X1*X1' - 2*(X1*V1')*U' + 2*W'*X2*X2' - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*P1'*P1*W'*X1*X1' + 2*nu*P2'*P2*W'*X2*X2' + 2*Lw*W';
        %    W = W - L*Vw;
            
            %W=W./norm(W);
        %end 
        %W=W+.01*eye(100);
    end
        V12 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2* X12) + (beta* A * V2) + nu * P1 * W2*X12);
        %for i = 1:10
        V22 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2* X22) + (beta* transpose(A) * V12) + nu * P2*W2 * X22);    
        V12 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2* X12) + (beta* A * V22) + nu * P1 * W2*X12);
        %end    
        V22 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2* X22) + (beta* transpose(A) * V12) + nu * P2*W2 * X22);    

        D = zeros(n,n);
    for m = 1:n
    
        %v1 = P1*(X12(:,m));
        %v1 = (W2*X1(:,m));
        v1 = V12(:,m);
        for i = 1:n
            %v2 = P2*(X22(:,i));
            %v2 = (W2*X2(:,i));
            v2 = V22(:,i);
            D(m,i) = norm(((v1 -A* v2)));
        end

    end
    CMC(D,100);
    hold on;
end
    
