%CUHK01 + LOMO
%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

feaFile = 'cuhk01_lomo.mat';
%pcaFile = 'matlabPCA100.mat';

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

    Lu = 0.05;
    Lv = 0.2;
    La = 0.2;
    Lp = 0.2;

    nu = 1;
    beta = 1;

    n = 316;
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
