%CSPL Paper Replication 65% Accuracy
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'viper_lomo.mat';
pcaFile = 'matlabPCA100.mat';

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
clear descriptors

    Lu = 0.05*1;
    L = 10^-13;
    Lv = 0.2*1;
    La = 0.2*1;
    Lp = 0.2*1;
    Lw = 0.5*1;

    nu = 1*1;
    beta = 1*1;

    n = 316;
    d = 100;
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
    
    X1 = X(:, 1:316);
    X2 = X(:, 317:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');
    
    TestPCA = W' * TestSet';
    X12 = TestPCA(:, 1:316);
    X22 = TestPCA(:, 317:end);

    
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
    std = 1;
    K1 = zeros(n,n);
    for m = 1:n
        xi = X1(:,m);
        for i = 1:n
            xj = X1(:,i);
            %K1(m,i) = exp(-(((norm(xi - xj))^2)/std));
            K1(m,i) = xi'*xj;
        end
    end    
    
    
    
    K2 = zeros(n,n);
    for m = 1:n
        xi = X1(:,m);
        for i = 1:n
            xj = X2(:,i);
            %K2(m,i) = exp(-(((norm(xi - xj))^2)/std));
            K2(m,i) = xi'*xj;
        end
    end    
    
    
    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P2 = randi([0, 1], [k,d]);
    W2 = eye(k);
    %Z = randi([0, 1], [n,d]);
    Z = eye(n,d);
    %Z = zeros(n,d);



%% Main algorithm
    for i = 1:1500
        U  = (( Z'*K1 * transpose(V1)) + ( Z'*K2 * transpose(V2)))/((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *Z'*K1) + (beta* A * V2) + nu * P1 * Z'*K1);
        V2 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U)* Z'*K2) + (beta* transpose(A) * V1) + nu * P2* Z'*K2);
        P1 = (V1 * transpose(Z'*K1)) / ((Z'*K1 * transpose(Z'*K1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(Z'*K2)) /((Z'*K2 * transpose(Z'*K2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) /((V2 * transpose(V2)) + (La/beta)*eye(k));
        %Z  = (K1*K1' + K2*K2' + K1) \ (K1*V1'*U' + K2*V2'*U');
        %W2 = inv((2*X1*X1' + 2*X2*X2' + Lw*eye(k)))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = inv((4*X1*X1' + 4*X2*X2' + Lw*eye(k) -X1*X2' -X2*X1'))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = W2';

        %for j = 1:300           
        %    Vw = 4*W2'*X1*X1' + 4*W2'*X2*X2' - 2*(X1*V1')*U'  - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*P1'*P1*W2'*X1*X1' + 2*nu*P2'*P2*W2'*X2*X2' + 2*Lw*W2' - X1*X2' - X2*X1';
        %    W2 = W2 - L*Vw;
        %     Vz = 2*(K1*K1')*Z -2*(K1*V1'*U' + K2*V2'*U') + 2*(K2*K2')*Z + 2*K1*Z -2*(K1*V1'*P1' + K2*V2'*P2') + 2*(K2*K2')*Z*(P2'*P2) + 2*(K1*K1')*Z*(P1'*P1);
        %     Z = Z - L*Vz;
            %W=W./norm(W);
        %end 
        %W2 = W2';
        
    end


    C1 = zeros(n,1);
    C12 = zeros(d,n);
    for m = 1:n
        xi = X12(:,m);
        for i = 1:n
            xj = X1(:,i);
            %C1(i) = exp(-(((norm(xi - xj))^2)/std));
            C1(i) = xi'*xj;
            
        end
        C12(:,m) = P1*Z'*C1;
    end  
    
    
    C2 = zeros(n,1);
    C22 = zeros(d,n);
    for m = 1:n
        xi = X22(:,m);
        for i = 1:n
            xj = X1(:,i);
            %C2(i) = exp(-(((norm(xi - xj))^2)/std));
            C2(i) = xi'*xj;
        end
        C22(:,m) = P2*Z'*C2;
    end 
    
    
%Final Distance computation
    D = 999*ones(n,n);    
    for m = 1:n
        xi0 = C12(:,m);
        for i = 1:n
            xj0 = C22(:,i);
            D(m,i) = norm(((xi0 - A*xj0)));
        end
    end
    
CMC(D,100);
