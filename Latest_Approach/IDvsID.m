%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'viper_lomo.mat';
numClass = 632;
numFolds = 2;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probeFea = descriptors(numClass + 1 : end, :);
clear descriptors


%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
%% Training 

    L = 0.000001;
    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;
    Lw = 0.1;

    nu = 0;
    beta = 200;

    n = 316;
    d = 100;
    k = d;

%for i = 1:10    
    p = randperm(numClass);
    
    galFea1 = galFea(p(1:numClass/2), : );
    probeFea1 = probeFea(p(1:numClass/2), : );

    TrainSet = zeros(632,26960);
    TrainSet(1:316 ,:) = galFea1;
    TrainSet(317: end,:) = probeFea1;
    
    t0 = tic;

    trainTime = toc(t0);

    [X , W] = matlabPCA(TrainSet',100);
    %load(pcaFile, 'X');
    %load(pcaFile, 'W');
    
    %X2 = X(1:15, 1:316);
    %X1 = X(1:15, 317:end);
    X2 = X(:, 1:316);
    X1 = X(:, 317:end);
    %X1 = pca(probeFea1');
    %X2 = pca(galFea1');
    
%    TestPCA = W' * TestSet';


    [W_XQDA, M] = XQDA(X1', X2', (1:numClass/2)', (1:numClass/2)');    
%    X1 = 3.* randi([0, 1], [d,n]);
%    X2 = X1 + 3;
%    X1 = X1 - 3;


%    X12 = 3.* randi([0, 1], [d,n]);
%    X22 = X12 + 3;
%    X12 = X12 - 3;
    
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

    %W = W(:,1:100);
%for i = 1:numFolds
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
        

        %W2 = W1';
        for j = 1:50
            Vw = 2*W2'*X1*X1' - 2*(X1*V1')*U' + 2*W2'*X2*X2' - 2*(X2*V2')*U' - 2*nu*X1*V1'*P1' - 2*nu*X2*V2'*P2' + 2*nu*P1'*P1*W2'*X1*X1' + 2*nu*P2'*P2*W2'*X2*X2' + 2*Lw*W2';
            W2 = W2 - L*Vw;
            
            %W=W./norm(W);
        end 
        %W=W+.01*eye(100);
    end
    

%% Testing

    galFea2 = galFea(p(numClass/2+1 : end), : );
    probeFea2 = probeFea(p(numClass/2+1 : end), : );
    
    TestSet = zeros(632,26960);
    TestSet(1:316 ,:) = galFea2;
    TestSet(317: end,:) = probeFea2;
    
    TestPCA = W' * TestSet';
    %TestPCA = W' * TestSet';
    %X22 = TestPCA(1:15, 1:316);
    %X12 = TestPCA(1:15, 317:end);
    X22 = TestPCA(:, 1:316);
    X12 = TestPCA(:, 317:end);
    
%    p1 = randperm(size(X22,2));
%    X22 = X22(:,p1);
    
    

    D = zeros(n,n);
    for m = 1:n

        %v1 = V12(:,m);
        for i = 1:n
            %v2 = P2*(X22(:,i));
            %v2 = (W2*X2(:,i));
            %v2 = V22(:,i);
            
            v1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) *W2* X12(:,m)) + (beta* A * V2) + nu * P1 * W2*X12(:,m));
            v2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) *W2* X22(:,i)) + (beta* transpose(A) * v1) + nu * P2*W2 * X22(:,i));    
            v1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) *W2* X12(:,m)) + (beta* A * v2) + nu * P1 * W2*X12(:,m));    
            v2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) *W2* X22(:,i)) + (beta* transpose(A) * v1) + nu * P2*W2 * X22(:,i));    
            D(m,i) = norm(((v1 -A* v2)));
        end

    end


    
    
    cms(nf,:) = EvalCMC( -D,1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
    
