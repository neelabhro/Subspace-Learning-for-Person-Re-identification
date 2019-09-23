%QMUL GRID + LOMO
%Neelabhro Roy
%IIIT-Delhi
                                                                                                                                                            
clear;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clc;                                                                                                            
close all;

feaFile1 = 'probe.mat';
feaFile2 = 'gallery.mat';
%pcaFile = 'CUHK01_LOMO_XQDA.mat';
pcaFile  = 'qmul.mat';

numClass = 125;
numFolds = 10;
numRanks = 100                                      ;                                                                     

%% load the extracted LOMO features
load(feaFile1, 'probe');
load(feaFile2, 'gallery');
galFea = gallery(:,(776:1025));
probeFea = probe;

p = randperm(numClass);

galFea1 = galFea(:, (1:125));
galFea2 = galFea(:, (126:250));
galFea3 = gallery(:,1:775);

probeFea1 = probeFea(:, (1:125));
probeFea2 = probeFea(:, (126:250));

    Lu = 0.05*1;
    L = 0.00000000001;
    Lv = 0.2*1;
    La = 0.2*1;
    Lp = 0.2*1;
    Lw = 0.5*1;

    nu = 1*1;
    beta = 1*1;

    n = 125;
    d = 100;
    k = d;

    p = randperm(numClass);

    TrainSet = zeros(250,26960);
    TrainSet(1:125,:) = galFea1';
    TrainSet(126: end,:) = probeFea1';
    
    t0 = tic;

    trainTime = toc(t0);

    TestSet = zeros(250,26960);
    TestSet1 = zeros(1025,26960);
    TestSet1(1:125 ,:) = galFea2';
    TestSet1(126: 250,:) = probeFea2';
    TestSet1(251:1025,:) = galFea3';
    
    %[X , W] = matlabPCA(TrainSet',100);
    load(pcaFile, 'X');
    load(pcaFile, 'W');
    
    X2 = X(:, 1:125);
    X1 = X(:, 126:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');
    
    TestPCA = W' * TestSet1';
    X22 = ones(100,900);
    X22(:,1:125) = TestPCA(:, 1:125);
    X22(:,126:900) = TestPCA(:,251:1025);
    X12 = TestPCA(:, 126:end);

    
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
    %P1 = eye(k);
    P2 = randi([0, 1], [k,d]);
    %P2 = eye(k);
    W2 = eye(k);
    



%% Main algorithm
    for i = 1:500
        U  = (( W2'*X1 * transpose(V1)) + ( W2'*X2 * transpose(V2)))/((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2'* X1) + (beta* A * V2) + nu * P1 * W2'*X1);
        V2 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2'* X2) + (beta* transpose(A) * V1) + nu * P2*W2' * X2);
        P1 = (V1 * transpose(W2'*X1)) / ((W2'*X1 * transpose(W2'*X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(W2'*X2)) /((W2'*X2 * transpose(W2'*X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) /((V2 * transpose(V2)) + (La/beta)*eye(k));
        %W2 = inv((2*X1*X1' + 2*X2*X2' + Lw*eye(k)))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = inv((4*X1*X1' + 4*X2*X2' + Lw*eye(k) -X1*X2' -X2*X1'))*(X1*V1'*U' + X2*V2'*U' + X1*V1'*P1 + X2*V2'*P2);
        %W2 = W2';

        for j = 1:100          
            Vw = 4*X1*X1'*W2 + 4*X2*X2'*W2 - 2*(X1*V1')*U'  - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*X1*X1'*W2*P1'*P1 + + 2*nu*X2*X2'*W2*P2'*P2 + 2*Lw*W2 - X1*X2' - X2*X1';
            W2 = W2 - L*Vw;           
            %W=W./norm(W);
        end 
        %W2 = W2';
        
    end

    
    D = 999*ones(n,n);
    for m = 1:n
    
        v1 = P1*W2'*(X12(:,m));
    
        for i = 1:n
            v2 = P2*W2'*(X22(:,i));
            D(m,i) = norm(((v1 - A*v2)));
        end
        
    end
    
CMC(D,100);
