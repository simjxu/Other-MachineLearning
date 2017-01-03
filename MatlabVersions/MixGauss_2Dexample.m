%% Get Data, Make Guesses
clear
TrainMatrix = ...
[0,0;0,1;1,0;1,1;4,4;4,5;5,4;5,5;8,8;8,9;9,8;9,9;];

% Make guesses on the Mu, Sigma, and Phi. Mu can be guessed using k-means
% algorithm
Mu = [0 0;4,4;8,8;];
Sigma(:,:,1) = [1,0.1;0.1,1];
Sigma(:,:,2) = [1,0.1;0.1,1];
Sigma(:,:,3) = [1,0.1;0.1,1];
Phi = [1/3;1/3;1/3;];

% Indicate number of iterations to make on the E-M algorithm
numiterations=1;

% Run E-M algorithm
numclusters = size(Mu,1);
numfeatures = size(TrainMatrix,2);
numtrainexamples = size(TrainMatrix,1);

w = zeros(numclusters,numtrainexamples);

% Begin while loop (following procedure in page 2-3 of the Mixture of
% Gaussians and EM algorithm section
iter = 0;
while iter < numiterations
    
    % E-step: determine the w-array
    PDF = zeros(numclusters,1);
    for i=1:numtrainexamples
        % E-step
        denom = 0;
        for n=1:numclusters
%             TrainMatrix(i,:)
%             Mu(n,:)
%             Sigma(:,:,n)
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
            TrainMatrix(i,:)
            Sigma(:,:,n)
        end
        % Divide by the total number of each category
        Sigma(:,:,n) = Sigma(:,:,n)/sum(w(n,:));
    end
    
    iter = iter+1;    
end

%% Plot scatter plot, and contour
scatter(TrainMatrix(:,1),TrainMatrix(:,2))
hold on

% Create contour
for l=1:3
    zz=gmdistribution(Mu(l,:),Sigma(:,:,l),Phi(l));
    ezcontour(@(x,y)pdf(zz,[x y]),[0 10],[0 10],250);
end
