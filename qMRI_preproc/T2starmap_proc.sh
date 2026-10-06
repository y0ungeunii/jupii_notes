## Step1: Denosing
dwidenoise ${input} ${output} -nthreads ${threads}
mrconvert ${denoised_img} -coord 3 0 -axes 0,1,2 ${echo1_denoised}.nii
mrconvert ${denoised_img} -coord 3 1 -axes 0,1,2 ${echo2_denoised}.nii
mrconvert ${denoised_img} -coord 3 2 -axes 0,1,2 ${echo3_denoised}.nii
mrconvert ${denoised_img} -coord 3 3 -axes 0,1,2 ${echo4_denoised}.nii
mrconvert ${denoised_img} -coord 3 4 -axes 0,1,2 ${echo5_denoised}.nii

## Step2: B1 correction

## Step3: B0 correction
N4BiasFieldCorrection  -d 3 -i ${input} -r -o ${output} -v

## Step4: Registration
# Registration echo1 to echoCombined image
echo1=sub-PNC019_ses-a1_acq-aspire_echo-1_part-mag_T2starw.nii.gz
comb=sub-PNC019_ses-a1_acq-aspire_desc-echoCombinedSensitivityCorrected_T2starw.nii.gz
antsRegistrationSyN.sh -d 3 -m ${echo1} -f ${comb} -o reg_test/sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_ -t s -n 150 -p d

# Registration echo2 to echoCombined image
echo2=sub-PNC019_ses-a1_acq-aspire_echo-2_part-mag_T2starw.nii.gz
antsRegistrationSyN.sh -d 3 -m ${echo2} -f ${echo1} -o reg_test/sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_ -t s -n 150 -p d
antsApplyTransforms -d 3 -i ${echo2} -r ${comb} -v -u int -o sub-PNC019_ses-a1_acq-aspire_echo-2TOechoCombined_T2starw.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_0GenericAffine.mat

# Registration echo3 to echoCombined image
echo3=sub-PNC019_ses-a1_acq-aspire_echo-3_part-mag_T2starw.nii.gz
antsRegistrationSyN.sh -d 3 -m ${echo3} -f ${echo2} -o reg_test/sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_ -t s -n 150 -p d
antsApplyTransforms -d 3 -i ${echo3} -r ${comb} -v -u int -o sub-PNC019_ses-a1_acq-aspire_echo-3TOechoCombined_T2starw.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_0GenericAffine.mat

# Registration echo4 to echoCombined image
echo4=sub-PNC019_ses-a1_acq-aspire_echo-4_part-mag_T2starw.nii.gz
antsRegistrationSyN.sh -d 3 -m ${echo4} -f ${echo3} -o reg_test/sub-PNC019_ses-a1_acq-aspire_echo-4TO3_T2starw_ -t s -n 200 -p d
antsApplyTransforms -d 3 -i ${echo4} -r ${comb} -v -u int -o sub-PNC019_ses-a1_acq-aspire_echo-4TOechoCombined_T2starw.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-4TO3_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-4TO3_T2starw_0GenericAffine.mat

# Registration echo5 to echoCombined image
echo5=sub-PNC019_ses-a1_acq-aspire_echo-5_part-mag_T2starw.nii.gz
antsRegistrationSyN.sh -d 3 -m ${echo5} -f ${echo4} -o reg_test/sub-PNC019_ses-a1_acq-aspire_echo-5TO4_T2starw_ -t s -n 200 -p d
antsApplyTransforms -d 3 -i ${echo5} -r ${comb} -v -u int -o sub-PNC019_ses-a1_acq-aspire_echo-5TOechoCombined_T2starw.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-1TOechoCombined_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-2TO1_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-3TO2_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-4TO3_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-4TO3_T2starw_0GenericAffine.mat -t sub-PNC019_ses-a1_acq-aspire_echo-5TO4_T2starw_1Warp.nii.gz -t sub-PNC019_ses-a1_acq-aspire_echo-5TO4_T2starw_0GenericAffine.mat

## Step5: Model fitting
python /host/verges/tank/data/youngeun/git/MyRelax/myrelax/getT2T2star.py sub-PNC019_ses-a1_acq-aspire_ALLecho_T2starw_reg.nii ../echo_times.txt sub-PNC019_ses-a1_acq-aspire_T2starmap_ --algo nonlinear --ncpu 64 --mask ../sub-PNC019_ses-a1_acq-aspire_desc-echoCombinedSensitivityCorrected_T2starw_mask.nii.gz

## Step6: Run QC
sub=sub-PNC019
ses=ses-a1

sing_img=/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif
bids=/data_/mica3/BIDS_PNI/rawdata
fs_lic=/data_/mica1/01_programs/freesurfer-7.3.2/license.txt
out=/data_/mica3/BIDS_PNI/derivatives
tmpDir=/data/mica2/tmpDir
threads=150
T2starmap_dir=/host/verges/tank/data/youngeun/T2star_preproc/reg_test/MyRelax
reg=/host/verges/tank/data/youngeun/T2star_preproc


singularity run --writable-tmpfs --containall -B ${tmpDir}:/tmpDir -B ${bids}:/bids -B ${out}:/out -B ${fs_lic}:/opt/licence.txt -B ${T2starmap_dir}:/T2starmap_dir -B ${reg}:/reg ${sing_img} -bids /bids -out /out -threads $threads -tmpDir /tmpDir -fs_licence /opt/licence.txt -MPC -sub $sub -ses $ses -mpc_acq T2starmap_reg_test_myrelax -regSynth -reg_nonlinear -microstructural_img /T2starmap_dir/${sub}_${ses}_acq-aspire_T2starmap__TxyME.nii -QC_subj -nocleanup -microstructural_reg /reg/${sub}_${ses}_acq-aspire_desc-echoCombined_T2starw.nii.gz

