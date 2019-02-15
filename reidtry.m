%Neelabhro Roy
%IIIT-Delhi

clear;
clc;
close all;

%x1 = imread('VIPeR/cam_a/000_45.bmp');
%imshow(x1);
%figure;
%x2 = imread('VIPeR/cam_b/000_45.bmp');  
%imshow(x2);

numClass = 316;
numFolds = 10;
numRanks = 100;

%For loading the extracted LOMO features
load('/viper_lomo.mat', 'descriptors');
%Gallery Features
%galFea = descriptors(1 : numClass, 1:100);
galFea = descriptors(1 : numClass,:);
galFea = transpose(galFea);
%Probe Features
probeFea = descriptors(numClass*2 + 1 : numClass*3, :);
probeFea = transpose(probeFea);
%probFea = descriptors(numClass + 1 : end, 1:100);
clear descriptors
images = zeros(128,48,3,316,'uint8');
camA = dir(['VIPeR/cam_a/*.bmp']);
camB = dir(['VIPeR/cam_b/*.bmp']);

for i = 1:5
    if ~camA(i).isdir && strcmp(camA(i).name(end-2:end), 'bmp')
        images(:,:,:,i) = imread(['VIPeR/cam_a/' camA(i).name]);
        %imshow(images(:,:,:,i));
        %figure;
    end
end 

for i = 1:5
    if ~camB(i).isdir && strcmp(camB(i).name(end-2:end), 'bmp')
        images(:,:,:,i) = imread(['VIPeR/cam_b/' camB(i).name]);
        %imshow(images(:,:,:,i));
        %figure;
    end
end 
%X1 = images;

Lu = 0.05;
Lv = 0.2;
La = 0.2;
Lp = 0.2;

nu = 1;
beta = 1;

%probeFeaT = transpose(probeFea);
probe_PCA = pca(probeFea);
for m=1:216
    probe_PCA = pca(transpose(probe_PCA));
end
probe_PCA = transpose(probe_PCA);


gal_PCA = pca(galFea);
for m=1:216
    gal_PCA = pca(transpose(gal_PCA));
end
gal_PCA = transpose(gal_PCA);
%d = length(probe_PCA);
d = 100;
k = d;

%X1 = transpose(probFea);
%X2 = transpose(galFea);
X1 = probe_PCA;
X2 = gal_PCA;
n = 316;

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
P2 = randi([0, 1], [k,d]);

%V1T = V1';

%Main algorithm
U  = (( X1 * transpose(V1)) + ( X2 * transpose(V2))) .* inv((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
V1 = inv(((transpose(U) * U) + (nu + beta + Lv) * eye(k))) * ((transpose(U) * X1) + (beta* A * V2) + nu * P1 * X1);
V2 = inv(((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) * ((transpose(U) * X2) + (beta* transpose(A) * V1) + nu * P2 * X2);
P1 = (V1 * transpose(X1)) * inv((X1 * transpose(X1)) + (Lp/nu)*eye(k));
P2 = (V2 * transpose(X2)) * inv((X2 * transpose(X2)) + (Lp/nu)*eye(k));
A  = (V1 * transpose(V2)) * inv((V2 * transpose(V2)) + (La/beta)*eye(k));

%v1 = P1.*x1;
%v2 = P2.*x2;
%D  = abs((v1 - A*v2)^2);
