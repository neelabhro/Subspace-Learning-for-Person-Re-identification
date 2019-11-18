%CUHK01 + LOMO
%Neelabhro Roy
%IIIT-Delhi
                                                                                                                                                            
clear;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clc;                                                                                                            
close all;

feaFile = 'cuhk01_lomo.mat';
pcaFile = 'cuhk01100.mat';

numClass = 970;
numFolds = 10;
numRanks = 100                                      ;                                                                     

%% load the extracted LOMO features
load(feaFile, 'descriptors');
camA1 = descriptors(:, 1:4:end);
camA2 = descriptors(:, 2:4:end);
camB1 = descriptors(:, 3:4:end);
camB2 = descriptors(:, 4:4:end);
p = randperm(numClass);
A = 1:1:1940;

FirstElements = A(:, 1:4:end);
SecondElements = A(:, 2:4:end);
AllElementsMatrix = [FirstElements; SecondElements];
[Rows, Cols] = size(AllElementsMatrix);
Array = reshape(AllElementsMatrix, [1, Rows*Cols]);
galFea1 = descriptors(:, Array);

FirstElements = A(:, 3:4:end);
SecondElements = A(:, 4:4:end);
AllElementsMatrix = [FirstElements; SecondElements];
[Rows, Cols] = size(AllElementsMatrix);
Array = reshape(AllElementsMatrix, [1, Rows*Cols]);
probeFea1 = descriptors(:, Array);

%load(pcaFile, 'p');

%galFea1 = camA1(:, p(1:485));
galFea2 = camA1(:, p(486:970));
galFea3 = camA2(:, p(486:970));
galFea4 = (galFea3 + galFea2)./2;

%probeFea1 = camB1(:, p(1:485));
probeFea2 = camB1(:, p(486:970));
probeFea3 = camB2(:, p(486:970));
probeFea4 = (probeFea3 + probeFea2)./2;

    n = 970;
    d = 100;
    k = d;
    
    TrainSet = zeros(1940,35722);
    TrainSet(1:970 ,:) = galFea1';
    TrainSet(971: end,:) = probeFea1';
    
    TestSet = zeros(970,35722);
    TestSet(1:485 ,:) = galFea4';
    TestSet(486: end,:) = probeFea4';
    [X ,W] = matlabPCA(TrainSet',100);