%% Principal Components Analysis
clear
% Data matrix is mxn, where m is the number of samples, and n is the number
% of features
DataMatrix = [0 0; -1 -1; -2 -2; 1 1; 2 2];
% DataMatrix = [0 0; -2 -2; -1 -2; 2 2; 1 2];
% DataMatrix = [0 0; 5 1; -5 -1; 4 1; -3 0];
scatter(DataMatrix(:,1),DataMatrix(:,2))
hold on

%% Normalize mean and variance
numsamples = size(DataMatrix,1);
numfeatures = size(DataMatrix,2);

% Calculate mu
mu = zeros(1,numfeatures);
for j=1:numfeatures
    mu(j) = 1/numsamples*sum(DataMatrix(:,j));
end

% Initialize the normalized training matrix
DataMatrixnorm = zeros(numsamples, numfeatures);

% Zero out the mean on the normalized matrix
for i=1:numsamples
    DataMatrixnorm(i,:) = DataMatrix(i,:)-mu;
end

% Set data to unit variance
for j=1:numfeatures
    sigma = sqrt(1/numsamples*sum(DataMatrix(:,j).^2));
    for i=1:numsamples
        DataMatrixnorm(i,j) = DataMatrix(i,j)/sigma;
    end
end

%% PCA Algorithm

% Calculate covariance
Covar = zeros(numfeatures,numfeatures);
for i = 1:numsamples
    Covar = Covar + DataMatrixnorm(i,:)'*DataMatrixnorm(i,:);
end
Covar = 1/numsamples*Covar;

% Identify the eigenvectors and eigenvalues. The largest eigenvalues
% correspond to the eigenvector with the largest variance
[eigvec,eigdiag] = eig(Covar);
eigval = zeros(1,numfeatures);
for i = 1:numfeatures
    eigval(i) = eigdiag(i,i);
end

% Find maximum eigenvalue index
[~,idx] = max(eigval);

primaryvec = eigvec(:,idx);

%% Add in mean and Unscale
primevec_unsc = zeros(numfeatures,1);

% Primevec_unsc is the unscaled version, with the sigma multiplied
for j=1:numfeatures
    sigma = sqrt(1/numsamples*sum(DataMatrix(:,j).^2));
    primevec_unsc(j)=primaryvec(j)*sigma;
end

% Make a line plot to plot out the primary vector
vectorplotvalx = [min(min(DataMatrix)):0.1:max(max(DataMatrix))];
vectorplotvaly = vectorplotvalx*primevec_unsc(2)/primevec_unsc(1);

plot(vectorplotvalx,vectorplotvaly)

hold off
