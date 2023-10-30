close all
f1 = 90;
fs = 500;
n = 0:50;
t = n/fs;
a = cos(2*pi*f1*t);
stem(t,a)
%% ---------------
f_con = 10000;
end_n_con = floor(max(n)/fs*f_con);
n_con = 0:end_n_con;
t_con = n_con/f_con;
a_con = cos(2*pi*f1*t_con);
hold on 
plot(t_con,a_con)
%% ---------------
n_unif = n(1:5:end);
t_unif = n_unif/fs;
a_unif = cos(2*pi*f1*t_unif);
stem(t_unif,a_unif)
%% 
f1_aliased = 10;
a_con = cos(2*pi*f1_aliased*t_con);
hold on 
plot(t_con,a_con)
%% 
figure
rand_idx = sort(randperm(length(n),length(n_unif)));

n_rand = n(rand_idx);
a_rand = cos(2*pi*f1*n_rand/fs);
stem(n_rand,a_rand)