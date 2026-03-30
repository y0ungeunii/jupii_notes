function [] = biasCorrMTSatSet(subjectDir)
%INPUTS: 
%1) subjectDir: absolute path of subject's 7 T MRI directory 
subjectDir = "/data/mica3/BIDS_PNI/derivatives/MTs_for_release";
%subjectList = {'PNC003', 'PNC006', 'PNC007', 'PNC009', 'PNC010', 'PNC015', 'PNC016', 'PNC018', 'PNC019', 'PNC022'};
subjectList = {'PNC024', 'PNC026', 'PNE007', 'PNE009', 'PNE014'}

for i = 1:numel(subjectList)
    subject = subjectList{i};

    T1w = char(strcat(subjectDir, "/ses-a1/", subject, "/sub-", subject, "_ses-a2_acq-mtw_T1w.nii,1"));
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {T1w};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 1e-05;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];
    
    MTw = char(strcat(subjectDir, "/ses-a1/", subject, "/sub-", subject, "_ses-a2_acq-mtw_mt-on_MTR.nii,1"));
    matlabbatch{2}.spm.spatial.preproc.channel.vols = {MTw};
    matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 1e-05;
    matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{2}.spm.spatial.preproc.channel.write = [1 1];
    
    PDw = char(strcat(subjectDir, "/ses-a1/", subject, "/sub-", subject, "_ses-a2_acq-mtw_mt-off_MTR.nii,1"));
    matlabbatch{3}.spm.spatial.preproc.channel.vols = {PDw};
    matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 1e-05;
    matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{3}.spm.spatial.preproc.channel.write = [1 1];
    
    addpath('/data/mica1/01_programs/spm12');
    jobs = {matlabbatch{1},matlabbatch{2},matlabbatch{3}}; 
    spm('defaults', 'FMRI');
    spm_jobman('run', jobs);

end 

end