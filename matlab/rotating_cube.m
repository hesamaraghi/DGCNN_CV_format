%%
r_Z = 0.2;
gamma = asin(r_Z);
alpha_min = asin(r_Z/sqrt(2));
alpha = asin(r_Z/sqrt(2)):0.000001:asin(r_Z);
x_p = alpha +acos(sin(alpha)/r_Z)-asin(r_Z);
beta = asin(r_Z) - alpha;
x_n = -(x_p + 2*beta);
close all
plot(x_p/pi*180,alpha/pi*180)
hold on
plot(x_n/pi*180,alpha/pi*180)
plot(-x_n/pi*180,alpha/pi*180)
disp((-min(x_n)+max(x_p))/pi*180)
disp(gamma - alpha_min)
%%
x_min = -acos(sin(alpha_min)/sin(gamma))-(gamma-alpha_min);
x_max = acos(sin(alpha_min)/sin(gamma))-(gamma-alpha_min);
x = x_min:0.001:x_max;
alpha = atan(cos(x+gamma)./(1/sin(gamma)-sin(x+gamma)));
disp((x_max - x_min)*180/pi)
plot(x*180/pi,alpha*180/pi)
%%
vertical_resolution = 1000;
figure
r_Zs = 0.01:0.001:0.5;
ifov = asin(r_Zs) - asin(r_Zs/2^.5);
plot(r_Zs,ifov*vertical_resolution)