%%
% close all
delta = 1/3;
t_start = max(ts)*(0.5 - delta/2);
t_end = max(ts)*(0.5 + delta/2);
idx2 = ts < t_end & ts > t_start;
idx_p = idx2 & idx; 
idx_n = idx2 & (~idx);
disp(sum(idx2))

figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.')
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.')
hold off

% ax.XDir = 'reverse';
ax.YDir = 'reverse';
ax.ZDir = 'reverse';
t_surf_min = inf(max(x)+1,max(y)+1);
cc = 1;
for x_i = x(:)'
        t_surf_min(x_i+1,y(cc)+1) = min(t_surf_min(x_i+1,y(cc)+1),ts(cc));
        cc =cc + 1;
end
ts_new = ts;
cc = 1;
for x_i = x(:)'
        ts_new(cc) = ts_new(cc) - t_surf_min(x_i+1,y(cc)+1);
        cc =cc + 1;
end
figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.')
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.')
hold off

