%CUHK01 + LOMO
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'cuhk01_lomo.mat';
pcaFile = 'CUHK01_LOMO_XQDA.mat';

numClass = 485;
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


    L = 0.000001;
    Lu = 0.5;
    Lv = 2;
    La = 2;
    Lp = 2;
    Lw = 0.01;

    nu =0;
    beta = 200;
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
    
    load(pcaFile, 'X1','X2','X22','X12','W_XQDA');
    %[W_XQDA, M] = XQDA(X1', X2', (1:numClass/2)', (1:numClass/2)');
    
    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P1 = eye(k);
    P2 = randi([0, 1], [k,d]);
    P2 = eye(k);
    W2 = randi([0, 1], [d,k]);
    W2 = W_XQDA';
    %V12 = randi([0, 1], [k,n]);
    %V22 = randi([0, , [k,n]);


    
%% Main algorithm
    for i = 1:500
        U  = (( W2*X1 * transpose(V1)) + ( W2*X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) *W2* X1) + (beta* A * V2) + nu * P1 * W2*X1);
        V2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) *W2* X2) + (beta* transpose(A) * V1) + nu * P2*W2 * X2);
        %P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        %P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) /((V2 * transpose(V2)) + (La/beta)*eye(k));
        %W2 = (X1*V1'*U' + X2*V2'*U')*inv((X1*X1' + X2*X2' - Lw*eye(k)));
        
        %for j = 1:k
        %    W2(:,j) = W2(:,j)./norm(W2(:,j));
        %end
        


        for j = 1:50
            Vw = 2*W2'*X1*X1' + 2*W2'*X2*X2' - 2*(X1*V1')*U'  - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*P1'*P1*W2'*X1*X1' + 2*nu*P2'*P2*W2'*X2*X2' + 2*Lw*W2';
            W2 = W2 - L*Vw;
            

        end 

    end
    
    
%% Testing    
    
    D = zeros(n,n);
    for m = 1:n
    
        V12 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) *W2* X12) + (beta* A * V2) + nu * P1 * W2*X12);
        V22 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) *W2* X22) + (beta* transpose(A) * V12) + nu * P2*W2 * X22);    
        V12 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) *W2* X12) + (beta* A * V22) + nu * P1 * W2*X12);    
        V22 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) *W2* X22) + (beta* transpose(A) * V12) + nu * P2*W2 * X22);    

        v1 = V12(:,m);
        for i = 1:n
            v2 = V22(:,i);
            D(m,i) = norm(((v1 -A* v2)));
        end

    end
    CMC(D,100);
    %hold on;
%end    
    figure;
    
for i = 1:10
    x = 10;  %Number of Identities to check
    p1 = randperm(x);
    
    Y1 = tsne(V12(:,p1(1:x))','Algorithm','barneshut','NumDimensions',3);
    Y2 = tsne((A*V22(:,p1(1:x)))','Algorithm','barneshut','NumDimensions',3);
    Y5=[Y1;Y2];
    G=[1:x,1:x];
    scatter3(Y5(:,1),Y5(:,2),Y5(:,3),50,G,'filled');
    figure;
end    
