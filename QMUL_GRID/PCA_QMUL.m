%QMUL GRID + LOMO
%Neelabhro Roy
%IIIT-Delhi
                                                                                                                                                            
clear;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clc;                                                                                                            
close all;

feaFile1 = 'probe.mat';
feaFile2 = 'gallery.mat';
%pcaFile = 'CUHK01_LOMO_XQDA.mat';

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
    TestSet(1:125 ,:) = galFea2';
    TestSet(126: end,:) = probeFea2';
    [X ,W] = matlabPCA(TrainSet',100);
