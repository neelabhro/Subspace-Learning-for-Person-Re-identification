% consider an artificial data set of 100 variables (e.g., genes) and 10 samples
function [pc]= matlabPCA(data,p)
% n = 700; % NUMBER OF DATA POINTS IN EACH CLASS
% N = 5; % DIMENSIONALITY OF DATA (MUST BE 2 FOR DISPLAY TO WORK)
% 
% A = .2*rand(N,n);  % CLASS 1
% B = -.2*rand(N,n); % CLASS 2
% C = -.2*rand(N,n)-1; % CLASS 2
% data = [A B C];
% p = 2
% Example [datah xh]= matlabPCA(data,p);



origdata = data;

% remove the mean variable-wise (row-wise)
meanmat = repmat(mean(data,2),1,size(data,2));
data=data-meanmat;

% calculate eigenvectors (loadings) W, and eigenvalues of the covariance matrix
[W, EvalueMatrix] = eig(cov(data'));  %Get eigenvectors of centere covariance matrix
Evalues = diag(EvalueMatrix);

% order by largest eigenvalue
[~,ind] = sort(abs(Evalues), 'descend'); % arrange eigenvectors in descending order of eigenvalues

W = W(:,ind);    
X = W;
%
W=W(:,1:p);  %%%%%5choose only p eigenvectors to retain p dim

% generate PCA component space (PCA scores)
pc = W' * data;  %encodes in PCA space 

datah = pinv(W')*pc + meanmat;  % reconstruct the data

sum(sum(abs(origdata - datah)))   % find error between original and reconstruction

%%%%%%555Using Matlab
X = pca(data');  %inbuilt function to get X
y = X(:,1:p)'*data;
xh = pinv(X(:,1:p)')*y + meanmat;
sum(sum(abs(origdata - xh)))



% % plot PCA space of the first two PCs: PC1 and PC2
%     plot(pc(1,:),'.')  
%     
%     %PCA 2 use definition of covariance
%     
% %     covariance = (1/(size(data,2)-1))*(data*data');
%     covariance = (data*data');
%     [W1, EvalueMatrix] = eig(covariance);
%     Evalues = diag(EvalueMatrix);
% 
% % order by largest eigenvalue
%     Evalues = Evalues(end:-1:1);
%     W1 = W1(:,end:-1:1); 
%     W1=W1(:,1:p);  
% 
% % generate PCA component space (PCA scores)
%     pc1 = W1' * data;
%     
%     datah1 = pinv(W1')*pc1 + meanmat;
%     
%     
%     % PCA3: Perform PCA using SVD. 
% % data - MxN matrix of input data 
% % (M dimensions, N trials) 
% % signals - MxN matrix of projected data 
% % PC - each column is a PC 
% % V - Mx1 matrix of variances 
% 
% % construct the matrix Y 
% Y = data / sqrt(size(data,2)-1); 
% % SVD does it all 
% [u,S,PC] = svd(Y); 
% % calculate the variances 
% S = diag(S); 
% V = S .* S; 
% % project the original data 
% signals = u' * data; 
% invsig = u*signals+meanmat;
