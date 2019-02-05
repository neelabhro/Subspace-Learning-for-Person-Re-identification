%Neelabhro Roy
%IIIT-Delhi
clear;
clc;
close all;

X9 = imread('VIPeR/cam_a/000_45.bmp');
imshow(X9);
figure;

X8 = imread('VIPeR/cam_b/000_45.bmp');
imshow(X8);

for m=1:2
  probe{m} = imread(sprintf('VIPeR/cam_a/%03d_45.bmp',m));
end

for m=1:2
  gallery{m} = imread(sprintf('VIPeR/cam_b/%03d_90.bmp',m));
end

Lu = 0.05;
Lv = 0.2;
La = 0.2;
Lp = 0.2;

nu = 1;
beta = 1;

d = 100;
k = 100;

n = length(probe);

% n is the size of the sample set
% d is the feature dimension equal to 100
% Components of Probe set X1 and gallery set X2 are matched image descriptors
% K is the number of latent factors and is equal to d
% U is the basis matrix capturing latent intrinsic structure of the input
% data matrix
% V1 and V2 (kxn) indicate the semantic representation of X1 and X2

% X(dxn) = U(dxk)*V(kxn)
X1 = randi([0, 1], [d,n]);
X2 = randi([0, 1], [d,n]);

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
