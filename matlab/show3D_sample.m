% clear all 
% close all
beta = 1e-5;
dataset_name = 'fan1vs3';
main_addr = '../datasets';

if(strcmpi(dataset_name,'fan1vs3') && ~exist('file_list_fan1vs3','var'))
    DS_folder = [main_addr '/fan1vs3/downloaded'];
    class_folder = [DS_folder '/segmented']; 
    file_list_fan1vs3 = getCategorizedPaths(class_folder);
    class_names_fan1vs3 = keys(file_list_fan1vs3);
end

if(strcmpi(dataset_name,'ncaltech101') && ~exist('file_list_ncaltech101','var'))
    DS_folder = [main_addr '/NCALTECH101/downloaded'];
    class_folder = [DS_folder '/Caltech101/']; 
    file_list_ncaltech101 = getCategorizedPaths(class_folder);
    class_names_ncaltech101 = keys(file_list_ncaltech101);
end

if(strcmpi(dataset_name,'nasl') && ~exist('file_list_nasl','var'))
    class_folder = [main_addr '/NASL/downloaded/'];
    file_list_nasl = getCategorizedPaths(class_folder);
    class_names_nasl = keys(file_list_nasl);
end

if(strcmpi(dataset_name,'ncars') && ~exist('file_list_ncars','var'))
    class_folder = [main_addr '/NCARS/downloaded/n-cars_test/']; 
    file_list_ncars = getCategorizedPaths(class_folder);
    class_names_ncars = keys(file_list_ncars);
end



if(strcmp(dataset_name,'nasl'))
    class_id = randi(length(class_names_nasl));
    class_name = class_names_nasl{class_id};
    sample_id = randi(length(file_list_nasl(class_name)));
    class_files = file_list_nasl(class_name);
    file = class_files{sample_id};
    file = ['/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NASL/downloaded/l/l_2245.mat'];
    load(file)
    idx = pol == 1;
elseif(strcmp(dataset_name,'ncaltech101'))
    class_id = randi(length(class_names_ncaltech101));
    class_name = class_names_ncaltech101{class_id};
    sample_id = randi(length(file_list_ncaltech101(class_name)));
    class_files = file_list_ncaltech101(class_name);
    file = class_files{sample_id};
    
    file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NCALTECH101/downloaded/Caltech101/lobster/image_0030.bin';

    TD_ncaltech101 = Read_Ndataset(file);
    x = TD_ncaltech101.x;
    y = TD_ncaltech101.y; 
    ts = TD_ncaltech101.ts;
    idx = TD_ncaltech101.p == 1;
elseif(strcmp(dataset_name, 'ncars'))
    class_id = randi(length(class_names_ncars));
    class_name = class_names_ncars{class_id};
    sample_id = randi(length(file_list_ncars(class_name)));
    class_files = file_list_ncars(class_name);
    disp(class_name)
    file = class_files{sample_id};
    
        file = '/Volumes/SFTP/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-GNN/datasets/NCARS/downloaded/n-cars_test/cars/obj_002262_td.dat';

    TD_ncars = load_atis_data(file);
    x = TD_ncars.x; 
    y = TD_ncars.y; 
    ts = TD_ncars.ts;
    idx = TD_ncars.p > 0;
elseif(strcmp(dataset_name, 'fan1vs3'))
    class_id = randi(length(class_names_fan1vs3));
    class_name = class_names_fan1vs3{class_id};
    sample_id = randi(length(file_list_fan1vs3(class_name)));
    class_files = file_list_fan1vs3(class_name);
    disp(class_name)
    file = class_files{sample_id};
    x = h5read(file,'/events/xs');
    y = h5read(file,'/events/ys');
    ts = h5read(file,'/events/ts');
    idx = strcmp(h5read(file,'/events/ps'),'TRUE');
end

%%
% close all

idx_p = idx; 
idx_n =  ~idx;


%%
X = zeros(1280,720);
for i = 1:length(ts)
    X(uint16(x(i))+1,uint16(y(i))+1) = X(uint16(x(i))+1,uint16(y(i))+1) + 1;
end
disp(max(X(:)))
[row,col] = find(X == max(X(:)));
disp([row,col])


%%
figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
hold off
ax.ZDir = 'reverse';
% axis equal
% ylim(ax,[0 240]);
% zlim(ax,[0 180]);

%%
num_ev = 1024;
subset_idx = randperm(length(ts),num_ev);
ts = ts(subset_idx);
x = x(subset_idx);
y = y(subset_idx);
idx_p = idx_p(subset_idx);
idx_n = idx_n(subset_idx);
%%


figure
ax = axes;
scatter3(ax,ts(idx_p)*1e-3,x(idx_p),y(idx_p),'r.','SizeData',100)
hold on
scatter3(ax,ts(idx_n)*1e-3,x(idx_n),y(idx_n),'b.','SizeData',100)
hold off
% % ax.YDir = 'reverse';
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

