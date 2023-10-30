close all
f1 = 90;
fs = 500;
n = 0:100;
n_unif = n(1:5:end);
%% 
N = 10000;
tensor_90 = zeros(2,length(n_unif),N);
for nn = 1:N
    rand_idx = sort(randperm(length(n),length(n_unif)));
    n_rand = n(rand_idx);
    t_rand = n_rand/fs;
    a_rand = cos(2*pi*f1*t_rand);
    tensor_90(1,:,nn) = n_rand;
    tensor_90(2,:,nn) = a_rand;
end
a_unif_90 = cos(2*pi*f1*n_unif/fs);
plot(n_unif/fs,a_unif_90)
%%
f1 = 110;
N = 10000;
tensor_110 = zeros(2,length(n_rand),N);
for nn = 1:N
    rand_idx = sort(randperm(length(n),length(n_unif)));
    n_rand = n(rand_idx);
    t_rand = n_rand/fs;
    a_rand = cos(2*pi*f1*t_rand);
    tensor_110(1,:,nn) = n_rand;
    tensor_110(2,:,nn) = a_rand;
end
hold on
a_unif_110 = cos(2*pi*f1*n_unif/fs);
plot(n_unif/fs,a_unif_110)
%%
matrix_90 = reshape(tensor_90,2*length(n_unif),N)';
matrix_110 = reshape(tensor_110,2*length(n_unif),N)';
data_matrix = [matrix_90;matrix_110];
labels = [zeros(N,1) ; ones(N,1)];
save('data.mat',"data_matrix")
save('labels.mat',"labels")
