%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'viper_lomo.mat';
pcaFile = 'matlabPCA100.mat';
wFile = 'PCA_W_XQDA.mat';

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
clear descriptors

    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;
    Lw = 0.1;

    nu = 1;
    beta = 1;

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
    
    %X2 = X(1:15, 1:316);
    %X1 = X(1:15, 317:end);
    X2 = X(:, 1:316);
    X1 = X(:, 317:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');
    
    TestPCA = W' * TestSet';
    %X22 = TestPCA(1:15, 1:316);
    %X12 = TestPCA(1:15, 317:end);
    X22 = TestPCA(:, 1:316);
    X12 = TestPCA(:, 317:end);

    
    
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
    load(wFile, 'W');
    W = W(:,1:100);
    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P2 = randi([0, 1], [k,d]);
    W1 = randi([0,1], [d,k]);
    %W1 = eye(k);
    W1 = W;
    W2 = randi([0,1], [d,k]);
    %W2 = eye(k);
    W2 = W;
    W1t = randi([0,1], [d,k]);
    W2t = randi([0,1], [d,k]);
    

%[V,D] = eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D


%% Main algorithm
    for i = 1:500
%        U  = (( W1'*X1 * transpose(V1)) + ( W2'*X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        U  = (( W1'*X1 * transpose(V1)) + ( W2'*X2 * transpose(V2))) / ((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = ((transpose(U) * U) + (nu + beta + Lv) * eye(k)) \ ((transpose(U) * W1'*X1) + (beta* A * V2) + nu * P1 *W1'* X1);
        V2 = ((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k)) \ ((transpose(U) *W2'* X2) + (beta* transpose(A) * V1) + nu * P2 * W2'*X2);
        P1 = (V1 * transpose(W1'*X1)) / ((W1'*X1 * transpose(W1'*X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(W2'*X2)) / ((W2'*X2 * transpose(W2'*X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) / ((V2 * transpose(V2)) + (La/beta)*eye(k));
        
        %A1 = X1*X1';
        %B1 = -Lw *inv(eye(k) + nu*P1'*P1);
        %C1 = (X1*V1'*W1 + nu*X1*V1'*P1)/(eye(k) + nu*P1'*P1);
                
        A1 = X1*X1';
        B1 = -Lw *inv(2*eye(k) + nu*P1'*P1);
        C1 = (X1*V1'*W1 + nu*X1*V1'*P1 + X1*X2'*W2)/(2*eye(k) + nu*P1'*P1)+0.0001*eye(k);
        
        [Ua1,da1] = eig(A1);
        [Vb1,db1] = eig(B1);
        C1t = ((Ua1 +eye(k)*0)\C1)*Vb1;
        
        for p = 1:d
        
            for q = 1:d
                W1t(p,q) = C1t(p,q) / ( da1(p,p) + db1(q,q) );
            end
            
        end
        %W1 = eye(k);
        W1 = W;
        %W1 = (Ua1*W1t) / (Vb1);
        for j = 1:k
            W1(j,:) = W1(j,:)./(norm(W1(j,:))+eps);
        end
        W1 = W;
        %W1 = eye(k);
        A2 = X2*X2';
        B2 = -Lw* inv(2*eye(k) + nu*P2'*P2);
        C2 = (X2*V2'*W2 + nu*X2*V2'*P2 + X2*X1'*W1)/(2*eye(k) + nu*P2'*P2)+0.0001*eye(k);        
        %A2 = X2*X2';
        %B2 = -Lw* inv(eye(k) + nu*P2'*P2);
        %C2 = (X2*V2'*W2 + nu*X2*V2'*P2)/(eye(k) + nu*P2'*P2);
        [Ua2,da2] = eig(A2);
        [Vb2,db2] = eig(B2);
        C2t = ((Ua2 +eye(k)*0)\C2)*Vb2;
        
        for p = 1:d
        
            for q = 1:d
                W2t(p,q) = C2t(p,q) / ( da2(p,p) + db2(q,q) );
            end
            
        end          
        %W2 = eye(k);
        W2 = (Ua2*W2t)/(Vb2);
        W2 = W;
        for j = 1:k
            W2(j,:) = W2(j,:)./(norm(W2(j,:))+eps);
        end
        %W2 = eye(k);
        W2 = W;        
    end

    
    D = zeros(n,n);
    for m = 1:n
    
        v1 = P1*(W1'*X1(:,m));
    
        for i = 1:n
            v2 = P2*(W2'*X2(:,i));
            D(m,i) = norm(((v1 - A*v2)));
        end
        
    end
    
CMC(D,100);