close all; clear; clc;

addpath(genpath('/host/percy/local_raid/ke/toolbox/BrainSpace'))
addpath(genpath('/host/percy/local_raid/ke/toolbox/gift'))
addpath(genpath('/host/percy/local_raid/ke/toolbox/ENIGMA'))
addpath(genpath('/host/percy/local_raid/ke/toolbox/spm12'))
addpath(genpath('/host/verges/tank/data/youngeun/myprogram/dpabi'))
%addpath(genpath('/data/mica1/01_programs/dpabi'))
%addpath(genpath('/host/verges/tank/data/youngeun/ReHo/neighbors2/fsLR32k'))

data_dir = '/data/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/';

% subject info
subjlist = readtable('Sublist_PNI.xlsx');
%subjID   = {'sub-PNC003', 'sub-PNC006', 'sub-PNC007', 'sub-PNC009', 'sub-PNC010', 'sub-PNC015', 'sub-PNC016', 'sub-PNC018', 'sub-PNC019', 'sub-PNC022'};
%subjID = {'sub-PNC003', 'sub-PNC006', 'sub-PNC011', 'sub-PNC018', 'sub-PNC019', 'sub-PNC022'};
%subjID = {'sub-PNC024'};
subjID = {'sub-PNC025', 'sub-PNC026', 'sub-PNC037', 'sub-PNC038'};
%site     = subjlist.site;

% mics: [1:52 305:404]
% epic: [58:88 405:441]
% nkg1: [89:127 442:498]
% nkg2: [128:224 499:563]

% parameters
neighbors = 1;
lh_mask   = '/host/verges/tank/data/youngeun/fsLR32k.L.mask.shape.gii';
rh_mask   = '/host/verges/tank/data/youngeun/fsLR32k.R.mask.shape.gii';
lh_surf   = '/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.L.surf.gii';
rh_surf   = '/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.R.surf.gii';
reho_path = '/host/verges/tank/data/youngeun/ReHo/neighbors1/fsLR32k/';

for i = 1:length(subjID)
    disp(['Processing subject: ' subjID{i}]);
    
    ts_name = strcat(data_dir, subjID(i), '/ses-a1/func/desc-me_task-rest_bold/surf/',...
                     subjID(i), '_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii'); % NOEL
    ts_img     = gifti(char(ts_name));
    ts_data    = ts_img.cdata; % timepoints x 64984
    ts_data_lh = ts_data(:, 1:end/2)';
    ts_data_rh = ts_data(:, end/2+1:end)';
    
    % compute vertex-wise ReHo
    lh_reho = y_reho_Surf(ts_data_lh, neighbors, lh_mask, lh_surf, 1);
    rh_reho = y_reho_Surf(ts_data_rh, neighbors, rh_mask, rh_surf, 1);

    % fsLR-32k -> schaefer-400
    %labeling     = load_parcellation('schaefer', 400);
    %labeling     = labeling.schaefer_400;
    lr_reho      = [lh_reho; rh_reho];
    %reho_schaefer = full2parcel(lr_reho', labeling)'; % 1 x 360
    output_file = strcat(reho_path, subjID{i}, '_ses-a1_surf-fsLR-32k_desc-ReHo.csv');
    %csvwrite(output_file, reho_schaefer);
    csvwrite(output_file, lr_reho);
    
    % save file
    %lr_reho = [lh_reho; rh_reho];
    %output_file = strcat(reho_path, subjID{i}, '_ses-03_surf-fsLR-32k_desc-ReHo.csv');
    %csvwrite(output_file, lr_reho);
end