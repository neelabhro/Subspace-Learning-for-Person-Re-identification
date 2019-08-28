%CUHK01 + LOMO + CSPL
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'cuhk01_lomo.mat';
pcaFile = 'CUHK01_A1_B1.mat';
%pcaFile = 'CUHK01_A2_B2.mat';

numClass = 970;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
camA1 = descriptors(:, 1:4:end);
camA2 = descriptors(:, 2:4:end);
camB1 = descriptors(:, 3:4:end);
camB2 = descriptors(:, 4:4:end);


galFea1 = camA1(:, 1:485);
galFea2 = camA1(:, 486:970);

probeFea1 = camB1(:, 1:485);
probeFea2 = camB1(:, 486:970);


    Lu = 0.05*1;
    L = 0.000000001;
    Lv = 0.2*1;
    La = 0.2*1;
    Lp = 0.2*1;
    Lw = 05*1;
    nu = 1;
    beta = 1;
    
    
    n = 485;
    d = 100;
    k = d;

    p = randperm(numClass);
    
%    galFea1 = galFea(p(1:numClass/2), : );
%    probeFea1 = probeFea(p(1:numClass/2), : );

    TrainSet = zeros(970,35722);
    TrainSet(1:485 ,:) = galFea1';
    TrainSet(486: end,:) = probeFea1';
    
    t0 = tic;

    trainTime = toc(t0);
%    galFea2 = galFea(p(numClass/2+1 : end), : );
%    probeFea2 = probeFea(p(numClass/2+1 : end), : );
    
    TestSet = zeros(970,35722);
    TestSet(1:485 ,:) = galFea2';
    TestSet(486: end,:) = probeFea2';
    
    load(pcaFile, 'X1','X2','X22','X12');
    %[W_XQDA, M] = XQDA(X1', X2', (1:numClass/2)', (1:numClass/2)');
    
    %X22 = X22(:,randperm(size(X22,2)));
    
    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P1 = eye(k);
    P2 = randi([0, 1], [k,d]);
    P2 = eye(k);
    W2 = eye(k);


    
%% Main algorithm
    for i = 1:500
        U  = (( W2*X1 * transpose(V1)) + ( W2*X2 * transpose(V2)))/((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2* X1) + (beta* A * V2) + nu * P1 * W2*X1);
        V2 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2* X2) + (beta* transpose(A) * V1) + nu * P2*W2 * X2);
        P1 = (V1 * transpose(X1)) / ((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(X2)) / ((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) / ((V2 * transpose(V2)) + (La/beta)*eye(k));
        %W2 = inv((2*X1*X1' + 2*X2*X2' + Lw*eye(k)))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = inv((4*X1*X1' + 4*X2*X2' + Lw*eye(k) -X1*X2' -X2*X1'))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = W2';
        %W2 = eye(k);
        %for j = 1:50
            
            %Vw = 4*W2'*X1*X1' + 4*W2'*X2*X2' - 2*(X1*V1')*U'  - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*P1'*P1*W2'*X1*X1' + 2*nu*P2'*P2*W2'*X2*X2' + 2*Lw*W2' - X1*X2' - X2*X1';
            %W2 = W2 - L*Vw;
            
            %W=W./norm(W);
        %end 
        %W2 = W2';
        
    end

    
    D = zeros(n,n);
    for m = 1:n
    
        v1 = P1*W2*(X12(:,m));
    
        for i = 1:n
            v2 = P2*W2*(X22(:,i));
            D(m,i) = norm(((v1 - A*v2)));
        end
        
    end
    
CMC(D,100);