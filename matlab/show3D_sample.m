clear all 
close all

dataset_name = 'dvs_gesture';
main_addr = '../datasets';


if(exist('file_list_nasl.mat','file')); load('file_list_nasl.mat'); end
if(exist('file_list_fan1vs3.mat','file')); load('file_list_fan1vs3.mat'); end
if(exist('file_list_ncaltech101.mat','file')); load('file_list_ncaltech101.mat'); end
if(exist('file_list_ncars.mat','file')); load('file_list_ncars.mat'); end
if(exist('file_list_dvsgesture.mat','file')); load('file_list_dvsgesture.mat'); end

if(strcmpi(dataset_name,'dvs_gesture') && ~exist('file_list_dvsgesture','var'))
    DS_folder = [main_addr '/DVS_GESTURE/test'];
    class_folder = DS_folder; 
    file_list_dvsgesture = getCategorizedPaths(class_folder);
    class_names_dvs_gesture = keys(file_list_dvsgesture);
    save 'file_list_dvsgesture.mat' file_list_dvsgesture class_names_dvs_gesture
end

if(strcmpi(dataset_name,'fan1vs3') && ~exist('file_list_fan1vs3','var'))
    DS_folder = [main_addr '/fan1vs3/downloaded'];
    class_folder = [DS_folder '/segmented']; 
    file_list_fan1vs3 = getCategorizedPaths(class_folder);
    class_names_fan1vs3 = keys(file_list_fan1vs3);
    save 'file_list_fan1vs3.mat' file_list_fan1vs3 class_names_fan1vs3
end

if(strcmpi(dataset_name,'ncaltech101') && ~exist('file_list_ncaltech101','var'))
    DS_folder = [main_addr '/NCALTECH101/downloaded'];
    class_folder = [DS_folder '/Caltech101/']; 
    file_list_ncaltech101 = getCategorizedPaths(class_folder);
    class_names_ncaltech101 = keys(file_list_ncaltech101);
    save 'file_list_ncaltech101.mat' file_list_ncaltech101 class_names_ncaltech101
end

if(strcmpi(dataset_name,'nasl') && ~exist('file_list_nasl','var'))
    class_folder = [main_addr '/NASL/downloaded/'];
    file_list_nasl = getCategorizedPaths(class_folder);
    class_names_nasl = keys(file_list_nasl);
    save 'file_list_nasl.mat' file_list_nasl class_names_nasl
end

if(strcmpi(dataset_name,'ncars') && ~exist('file_list_ncars','var'))
    class_folder = [main_addr '/NCARS/downloaded/n-cars_test/']; 
    file_list_ncars = getCategorizedPaths(class_folder);
    class_names_ncars = keys(file_list_ncars);
    save 'file_list_ncars.mat' file_list_ncars class_names_ncars
end



if(strcmp(dataset_name,'nasl'))
    camera_resolution = [240,180];
    class_id = randi(length(class_names_nasl));
    class_id = 1;
    class_name = class_names_nasl{class_id};
    sample_id = randi(length(file_list_nasl(class_name)));
    class_files = file_list_nasl(class_name);
    file = class_files{sample_id};
    % file = ['/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NASL/downloaded/l/l_2245.mat'];
    load(file)
    idx = pol == 1;
elseif(strcmp(dataset_name,'dvs_gesture'))



    camera_resolution = [180,180];
    class_id = randi(length(class_names_dvs_gesture));
    class_name = class_names_dvs_gesture{class_id};
    sample_id = randi(length(file_list_dvsgesture(class_name)));
    class_files = file_list_dvsgesture(class_name);
    file = class_files{sample_id};
    % file = ['/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NASL/downloaded/l/l_2245.mat'];
    load(file)
    ts = t;
    idx = p == 1;


elseif(strcmp(dataset_name,'ncaltech101'))
    camera_resolution = [240,180];
    class_id = randi(length(class_names_ncaltech101));
    class_name = class_names_ncaltech101{class_id};
    sample_id = randi(length(file_list_ncaltech101(class_name)));
    class_files = file_list_ncaltech101(class_name);
    file = class_files{sample_id};
    
    % file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NCALTECH101/downloaded/Caltech101/lobster/image_0030.bin';
    file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NCALTECH101/downloaded/Caltech101/butterfly/image_0001.bin';

    TD_ncaltech101 = Read_Ndataset(file);
    x = TD_ncaltech101.x;
    y = TD_ncaltech101.y; 
    ts = TD_ncaltech101.ts;
    idx = TD_ncaltech101.p == 1;
elseif(strcmp(dataset_name, 'ncars'))
    camera_resolution = [120,100];
    class_id = randi(length(class_names_ncars));
    class_name = class_names_ncars{class_id};
    sample_id = randi(length(file_list_ncars(class_name)));
    class_files = file_list_ncars(class_name);
    disp(class_name)
    file = class_files{sample_id};
    
        % file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NCARS/downloaded/n-cars_test/cars/obj_002262_td.dat';

    TD_ncars = load_atis_data(file);
    x = TD_ncars.x; 
    y = TD_ncars.y; 
    ts = TD_ncars.ts;
    idx = TD_ncars.p > 0;
elseif(strcmp(dataset_name, 'fan1vs3'))
    camera_resolution = [1280,720];
    class_id = randi(length(class_names_fan1vs3));

    class_id = 1;

    class_name = class_names_fan1vs3{class_id};
    sample_id = randi(length(file_list_fan1vs3(class_name)));
    class_files = file_list_fan1vs3(class_name);
    disp(class_name)
    file = class_files{sample_id};
    
    % file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/class_blink_0000.h5';
    % file = 'class_null_0000.hdf5';

    x = h5read(file,'/events/xs');
    y = h5read(file,'/events/ys');
    ts = h5read(file,'/events/ts');
    idx = strcmp(h5read(file,'/events/ps'),'TRUE');
end

%%
idx_p = idx; 
idx_n =  ~idx;
%%
figure
TD = struct('x', double(x)+1 ,  'y', double(y)+1,...
                'ts', double(ts), 'p', double(p));
ShowTD(TD);

%%
figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
hold off
% ax.ZDir = 'reverse';
ax.YDir = 'reverse';
%%
% close all

m = 3;
ev = [x(:),y(:),ts(:),idx(:)];
[ts,indices] = sort(ts(:),"ascend");
x = x(indices);
y = y(indices);
idx = idx(indices);
[locations,X_idx] = subsample_threshold(x,y,ts,idx,m);
x_th = x(locations);
y_th = y(locations);
ts_th = ts(locations);
idx_th = idx(locations);

idx_p_th = idx_th; 
idx_n_th =  ~idx_th;

figure
ax = axes;
scatter3(ax,ts_th(idx_p_th)*1e-3,x_th(idx_p_th),y_th(idx_p_th),'r.','SizeData',100)
hold on
scatter3(ax,ts_th(idx_n_th)*1e-3,x_th(idx_n_th),y_th(idx_n_th),'b.','SizeData',100)
hold off
ax.ZDir = 'reverse';
ax.YDir = 'reverse';

%%
X = zeros([max(x)+1,max(y)+1]);
for i = 1:length(ts)
    X(uint16(x(i))+1,uint16(y(i))+1) = X(uint16(x(i))+1,uint16(y(i))+1) + 1;
end
disp(max(X(:)))
[row,col] = find(X == max(X(:)));
disp([row,col])
figure
imagesc(X)
disp(nnz(X)/numel(X))
%%
% figure
% ax = axes;
% imagesc(ax,X')
% hold off
% ax.ZDir = 'reverse';
% ax.YDir = 'reverse';

%%
% figure
% ax = axes;
% scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
% hold on
% scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
% hold off
% ax.ZDir = 'reverse';
% xlim(ax,[7 8])
% axis equal
% ylim(ax,[0 240]);
% zlim(ax,[0 180]);

%%
% figure
% ax = axes;
% scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
% hold on
% scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
% hold off
% ax.ZDir = 'reverse';
% ax.YDir = 'reverse';

% axis equal
% ylim(ax,[0 240]);
% zlim(ax,[0 180]);

%% Revesrse the time plus polarity
% ts = max(ts) - ts; 

% figure
% ax = axes;
% scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'r.','SizeData',100)
% hold on
% scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'b.','SizeData',100)
% hold off
% ax.ZDir = 'reverse';
% xlim(ax,[3 4])
% axis equal
% ylim(ax,[0 240]);
% zlim(ax,[0 180]);
%%
num_ev = 4096;length(locations);
subset_idx = sort(randperm(length(ts),num_ev));
ts = ts(subset_idx);
x = x(subset_idx);
y = y(subset_idx);
idx_p = idx_p(subset_idx);
idx_n = idx_n(subset_idx);
%%
X = zeros([max(x)+1,max(y)+1]);
for i = 1:length(ts)
    X(uint16(x(i))+1,uint16(y(i))+1) = X(uint16(x(i))+1,uint16(y(i))+1) + 1;
end
disp(max(X(:)))
[row,col] = find(X == max(X(:)));
disp([row,col])
figure
imagesc(X)
disp(nnz(X)/numel(X))
%%


figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
hold off
ax.ZDir = 'reverse';
ax.YDir = 'reverse';
% ylim(ax,[0 240]);
% zlim(ax,[0 180]);


% % ax.XDir = 'reverse';
% ax.YDir = 'reverse';
% ax.ZDir = 'reverse';
% 
% figure
% ax = axes;
% scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.')
% hold on
% scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.')
% hold off

%%
% figure
% x_sample = 850;
% y_sample = 500;
% patch_length = 50000;
% ind_pixel = (abs(x - x_sample) <= patch_length) & (abs(y - y_sample) <= patch_length);
% p_sample = idx(ind_pixel);
% p_cumsum = cumsum(p_sample*2-1);
% ts_sample = ts(ind_pixel);
% plot(ts_sample,p_cumsum)

%%
