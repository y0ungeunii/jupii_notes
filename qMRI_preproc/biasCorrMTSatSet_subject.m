function [] = biasCorrMTSatSet_subject(subjectDir)
%INPUTS: 
%1) subjectDir: absolute path of subject's 7 T MRI directory
%subjectDir = "/data/mica3/BIDS_PNI/derivatives/B1-corrected-newT2star";
subjectDir = "/host/verges/tank/data/youngeun/T2star_preproc/denoise_B1_B0_reg_test";
%subjectList = {'PNC019'};
subjectList = {'1', '2', '3', '4', '5'};

for i = 1:numel(subjectList)
    subject = subjectList{i};

    %qMRI = char(strcat(subjectDir, "/", subject, "/sub-", subject, "_ses-a1_acq-aspire_T2starmap.nii,1"));
    qMRI = char(strcat(subjectDir, "/sub-PNC019_ses-a1_acq-aspire_echo-", subject, "_part-mag_T2starw_denoised.nii,1"));
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {qMRI};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 1e-05;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];
    
    addpath('/data/mica1/01_programs/spm12');
    jobs = {matlabbatch{1}}; 
    spm('defaults', 'FMRI');
    spm_jobman('run', jobs);

end

end