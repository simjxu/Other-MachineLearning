%% Get Data, Make Guesses
clear
TrainMatrix = ...
[0,0;0,1;1,0;1,1;4,4;4,5;5,4;5,5;8,8;8,9;9,8;9,9;];

% Guesses for Cluster centroids
Mu = [2 2; 4 3; 7 8];

%% k-means clustering algorithm
numsamples = size(TrainMatrix,1);
numclusters = size(Mu,1);

c = zeros(size(TrainMatrix));

for j = 1:numclusters
    for i = 1:numsamples
    end
end

%% Or, just use kmeans function

[idx,centroids] = kmeans(TrainMatrix,3)