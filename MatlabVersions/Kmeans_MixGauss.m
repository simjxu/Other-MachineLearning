function [Mu,Sigma,Phi] = Kmeans_MixGauss(TrainMatrix,numclusters, ...
    numiterations)
% TrainMatrix must be numsamples x numfeatures
% numclusters = 3; numiterations = 10;

% Identify number of features and number of training examples
numtrainexamples = size(TrainMatrix,1);
numfeatures = size(TrainMatrix,2);

% Set defaults for initial guesses ----------------
% Assume same prob for all clusters
Phi = ones(numclusters,1)*1/numclusters;
% Assume identity matrix for default Sigma
Sigma = zeros(numfeatures,numfeatures,numclusters);
for j = 1:numclusters
    for i = 1:numfeatures
        Sigma(i,i,j) = 1;
    end
end

% First, use kmeans to find the cluster centroids, and use those for means
[idx,centroids,sumD,D] = kmeans(TrainMatrix,numclusters);
Mu = centroids;

% Begin while loop (following procedure in page 2-3 of the Mixture of
% Gaussians and EM algorithm section
iter = 0;
w = zeros(numclusters,numtrainexamples);
while iter < numiterations
    
    % E-step: determine the w-array
    PDF = zeros(numclusters,1);
    for i=1:numtrainexamples
        % E-step
        denom = 0;
        for n=1:numclusters
            PDF(n) = mvnpdf(TrainMatrix(i,:),Mu(n,:),Sigma(:,:,n))*Phi(n);
            denom = denom+PDF(n);
        end

        % Calculate the w probability for each cluster
        w(:,i) = PDF/sum(PDF);
    end
    
    % M-step
    % Update phi for each category
    for n=1:numclusters
        Phi(n) = sum(w(n,:))/numtrainexamples;
    end
    
    % Update mu for each category
    Mu = zeros(numclusters,numfeatures);
    for n=1:numclusters
        for i=1:numtrainexamples
            Mu(n,:) = Mu(n,:) + w(n,i)*TrainMatrix(i,:);
        end
        % Divide by the total number of each category
        Mu(n,:) = Mu(n,:)/sum(w(n,:));
    end

    % Update Sigma for each category
    Sigma = zeros(numfeatures,numfeatures,numclusters);
    for n=1:numclusters
        for i=1:numtrainexamples
            Sigma(:,:,n) = Sigma(:,:,n) + w(n,i)* ...
                (TrainMatrix(i,:)-Mu(n,:))'*(TrainMatrix(i,:)-Mu(n,:));
        end
        Sigma(:,:,n) = Sigma(:,:,n)/sum(w(n,:));
    end

    iter = iter+1;    
end