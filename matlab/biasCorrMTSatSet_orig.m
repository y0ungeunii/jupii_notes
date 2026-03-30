function [] = biasCorrMTSatSet(subjectDir)
%INPUTS: 
%1) subjectDir: absolute path of subject's 7 T MRI directory 

T1w = char(strcat(subjectDir,"/NotRegist/MTSat_Image_Set/T1W_bfReg.nii,1")); 
matlabbatch{1}.spm.spatial.preproc.channel.vols = {T1w};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 1e-05;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 20;
matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];

MTw = char(strcat(subjectDir,"/NotRegist/MTSat_Image_Set/MTON_bfReg.nii,1")); 
matlabbatch{2}.spm.spatial.preproc.channel.vols = {MTw};
matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 1e-05;
matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 20;
matlabbatch{2}.spm.spatial.preproc.channel.write = [1 1];

PDw = char(strcat(subjectDir,"/NotRegist/MTSat_Image_Set/MTOFF_bfReg.nii,1")); 
matlabbatch{3}.spm.spatial.preproc.channel.vols = {PDw};
matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 1e-05;
matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 20;
matlabbatch{3}.spm.spatial.preproc.channel.write = [1 1];

addpath('/export02/data/risa/spm12');
jobs = {matlabbatch{1},matlabbatch{2},matlabbatch{3}}; 
spm('defaults', 'FMRI');
spm_jobman('run', jobs);

end 