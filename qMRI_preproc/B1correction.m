function biasCorrMTSatSet_subject(subjectDir, subjectList, sesList)

% B1 correction using SPM12
%
% INPUTS:
%   subjectDir  - Directory containing the qMRI data
%   subjectList - Cell array of subject IDs
%   sesList     - Cell array of session IDs
%
% Example:
%   subjectDir = '/path/to/data';
%   subjectList = {'sub-PNC001', 'sub-PNC002', 'sub-PNC003'};
%   sesList = {'ses-01', 'ses-02', 'ses-01'};
%
%   biasCorrMTSatSet_subject(subjectDir, subjectList, sesList);

%% Check inputs

if numel(subjectList) ~= numel(sesList)
    error('subjectList and sesList must have the same number of elements.');
end

%% Initialize SPM

addpath('/data/mica1/01_programs/spm12');
spm('defaults', 'FMRI');

%% B1 correction

for i = 1:numel(subjectList)

    subject = subjectList{i};
    ses = sesList{i};

    % Input qMRI image
    qMRI = fullfile(subjectDir, [subject '_' ses '_mt-on_MTR.nii,1']);

    % Check that input file exists
    if ~isfile(strrep(qMRI, ',1', ''))
        warning('Input file not found: %s', qMRI);
        continue;
    end

    % SPM preprocessing
    matlabbatch = [];
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {qMRI};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 1e-05;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];

    % Run
    spm_jobman('run', matlabbatch);

end
end

