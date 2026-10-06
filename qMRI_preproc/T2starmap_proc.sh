#!/usr/bin/env bash

###############################################################################
# T2* MAP PREPROCESSING PIPELINE
#
# Description:
#   Preprocess multi-echo T2*-weighted MRI data.
#
# Pipeline:
#   Step 1 - Denoising
#   Step 2 - B1 correction
#   Step 3 - B0 correction
#   Step 4 - Echo registration
#   Step 5 - T2* model fitting
#   Step 6 - Surface-based QC using micapipe
#
# Requirements:
#   - MRtrix3
#   - SPM12 / MATLAB
#   - ANTs
#   - MyRelax
#   - Python
#   - micapipe
#   - Singularity / Apptainer
#
###############################################################################


###############################################################################
# CONFIGURATION
#
# Modify this section for a new dataset / subject.
###############################################################################

# ----------------------------- Subject --------------------------------------

sub="sub-PNC019"
ses="ses-a1"

# Number of echoes
n_echoes=5

# ----------------------------- Directories ----------------------------------

# Input directory containing the original multi-echo T2* images
input_dir="/path/to/input"

# Main preprocessing directory
output_dir="/path/to/T2star_preproc"

# Registration output directory
reg_dir="${output_dir}/registration"

# T2* model fitting output directory
t2star_dir="${output_dir}/MyRelax"

# ----------------------------- Software -------------------------------------

# SPM12
spm_dir="/data/mica1/01_programs/spm12"

# MyRelax
myrelax="/host/verges/tank/data/youngeun/git/MyRelax/myrelax/getT2T2star.py"

# micapipe Singularity image
sing_img="/data/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif"

# ----------------------------- Computing ------------------------------------

threads=150

# ----------------------------- Echo information -----------------------------

# Echo times used for model fitting
echo_times="${output_dir}/echo_times.txt"

###############################################################################
# INPUT FILES
###############################################################################

# Combined echo image used as the registration target
combined="${output_dir}/${sub}_${ses}_acq-aspire_desc-echoCombinedSensitivityCorrected_T2starw.nii.gz"

# Combined echo mask
mask="${output_dir}/${sub}_${ses}_acq-aspire_desc-echoCombinedSensitivityCorrected_T2starw_mask.nii.gz"

# Denoised multi-echo image
denoised_img="${output_dir}/${sub}_${ses}_acq-aspire_desc-denoised_T2starw.nii.gz"


###############################################################################
# STEP 1: DENOISING
#
# Denoise the original multi-echo image using MRtrix3 dwidenoise.
#
# The resulting 4D image is then split into individual echo images.
###############################################################################

echo "=============================================="
echo "STEP 1: DENOISING"
echo "=============================================="

dwidenoise \
    "${input_dir}/${sub}_${ses}_acq-aspire_T2starw.nii.gz" \
    "${denoised_img}" \
    -nthreads "${threads}"


# Split individual echoes
for ((i=1; i<=n_echoes; i++)); do

    # MRtrix uses zero-based indexing
    echo_index=$((i-1))

    echo_file="${output_dir}/${sub}_${ses}_acq-aspire_echo-${i}_part-mag_T2starw_denoised.nii.gz"

    echo "Extracting echo ${i}..."

    mrconvert \
        "${denoised_img}" \
        -coord 3 "${echo_index}" \
        -axes 0,1,2 \
        "${echo_file}"

done


###############################################################################
# STEP 2: B1 CORRECTION
#
# B1 correction is performed using the MATLAB/SPM12 script.
#
# The MATLAB function should receive:
#   - subject directory
#   - subject ID
#   - session ID
#   - echo list
###############################################################################

echo "=============================================="
echo "STEP 2: B1 CORRECTION"
echo "=============================================="

# TODO:
# Add MATLAB call here once the MATLAB script is finalized.
#
# Example:
#
# matlab -batch "biasCorrMTSatSet_subject( \
#     '${output_dir}', \
#     {'${sub}'}, \
#     {'${ses}'} \
# )"


###############################################################################
# STEP 3: B0 CORRECTION
#
# Apply N4 bias field correction.
###############################################################################

echo "=============================================="
echo "STEP 3: B0 CORRECTION"
echo "=============================================="

N4BiasFieldCorrection \
    -d 3 \
    -i "${input_dir}/${sub}_${ses}_acq-aspire_T2starw.nii.gz" \
    -r \
    -o "${output_dir}/${sub}_${ses}_acq-aspire_T2starw_B0corrected.nii.gz" \
    -v


###############################################################################
# STEP 4: REGISTRATION
#
# Echoes are registered sequentially:
#
#   Echo 1 → Echo Combined
#   Echo 2 → Echo 1 → Echo Combined
#   Echo 3 → Echo 2 → Echo 1 → Echo Combined
#   ...
#
# This allows all echoes to be transformed into the space of the combined
# echo image.
###############################################################################

echo "=============================================="
echo "STEP 4: REGISTRATION"
echo "=============================================="

mkdir -p "${reg_dir}"


# Store transformation files for the previous echo
declare -a WARP_FILES
declare -a AFFINE_FILES


for ((i=1; i<=n_echoes; i++)); do

    echo "----------------------------------------------"
    echo "Registering echo ${i}"
    echo "----------------------------------------------"

    echo_file="${output_dir}/${sub}_${ses}_acq-aspire_echo-${i}_part-mag_T2starw.nii.gz"

    prefix="${reg_dir}/${sub}_${ses}_echo-${i}TO$((i-1))_T2starw_"


    ###########################################################################
    # Echo 1
    #
    # Echo 1 is directly registered to the combined echo image.
    ###########################################################################

    if [ "${i}" -eq 1 ]; then

        antsRegistrationSyN.sh \
            -d 3 \
            -m "${echo_file}" \
            -f "${combined}" \
            -o "${reg_dir}/${sub}_${ses}_echo-1TOechoCombined_T2starw_" \
            -t s \
            -n 150 \
            -p d

        WARP_FILES[1]="${reg_dir}/${sub}_${ses}_echo-1TOechoCombined_T2starw_1Warp.nii.gz"
        AFFINE_FILES[1]="${reg_dir}/${sub}_${ses}_echo-1TOechoCombined_T2starw_0GenericAffine.mat"

    ###########################################################################
    # Echoes 2+
    #
    # Each echo is registered to the previous echo.
    ###########################################################################

    else

        previous_echo="${output_dir}/${sub}_${ses}_acq-aspire_echo-$((i-1))_part-mag_T2starw.nii.gz"

        antsRegistrationSyN.sh \
            -d 3 \
            -m "${echo_file}" \
            -f "${previous_echo}" \
            -o "${prefix}" \
            -t s \
            -n 150 \
            -p d

        WARP_FILES[i]="${prefix}1Warp.nii.gz"
        AFFINE_FILES[i]="${prefix}0GenericAffine.mat"

    fi


    ###########################################################################
    # Apply all transformations
    #
    # Transformations are applied in reverse order, from the current echo
    # back through the previous echoes and finally to the combined image.
    ###########################################################################

    output_echo="${reg_dir}/${sub}_${ses}_acq-aspire_echo-${i}TOechoCombined_T2starw.nii.gz"

    transforms=()

    # Add transformations from current echo back to echo 2
    if [ "${i}" -gt 1 ]; then

        for ((j=i; j>=2; j--)); do
            transforms+=(-t "${WARP_FILES[j]}")
            transforms+=(-t "${AFFINE_FILES[j]}")
        done

    fi

    # Add Echo 1 → Echo Combined transformation
    transforms+=(-t "${WARP_FILES[1]}")
    transforms+=(-t "${AFFINE_FILES[1]}")


    antsApplyTransforms \
        -d 3 \
        -i "${echo_file}" \
        -r "${combined}" \
        -v \
        -u int \
        -o "${output_echo}" \
        "${transforms[@]}"

done


###############################################################################
# STEP 5: T2* MODEL FITTING
#
# Fit the T2* model using MyRelax.
###############################################################################

echo "=============================================="
echo "STEP 5: T2* MODEL FITTING"
echo "=============================================="

mkdir -p "${t2star_dir}"


python "${myrelax}" \
    "${output_dir}/${sub}_${ses}_acq-aspire_ALLecho_T2starw_reg.nii" \
    "${echo_times}" \
    "${t2star_dir}/${sub}_${ses}_acq-aspire_T2starmap_" \
    --algo nonlinear \
    --ncpu 64 \
    --mask "${mask}"


###############################################################################
# STEP 6: QC
#
# Run micapipe to register the T2* map to the subject's anatomical space
# and generate QC outputs.
###############################################################################

echo "=============================================="
echo "STEP 6: QC"
echo "=============================================="


# ----------------------------- micapipe paths -------------------------------

bids="/data/mica3/BIDS_PNI/rawdata"
out="/data/mica3/BIDS_PNI/derivatives"
fs_lic="/data/mica1/01_programs/freesurfer-7.3.2/license.txt"
tmpDir="/data/mica2/tmpDir"


# T2* map directory
T2starmap_dir="${t2star_dir}"


# Registration directory
reg="${output_dir}"


###############################################################################
# Run micapipe
###############################################################################

singularity run \
    --writable-tmpfs \
    --containall \
    -B "${tmpDir}:/tmpDir" \
    -B "${bids}:/bids" \
    -B "${out}:/out" \
    -B "${fs_lic}:/opt/licence.txt" \
    -B "${T2starmap_dir}:/T2starmap_dir" \
    -B "${reg}:/reg" \
    "${sing_img}" \
    -bids /bids \
    -out /out \
    -threads "${threads}" \
    -tmpDir /tmpDir \
    -fs_licence /opt/licence.txt \
    -MPC \
    -sub "${sub}" \
    -ses "${ses}" \
    -mpc_acq T2starmap_reg_test_myrelax \
    -regSynth \
    -reg_nonlinear \
    -microstructural_img "/T2starmap_dir/${sub}_${ses}_acq-aspire_T2starmap__TxyME.nii" \
    -QC_subj \
    -nocleanup \
    -microstructural_reg "/reg/${sub}_${ses}_acq-aspire_desc-echoCombined_T2starw.nii.gz"


###############################################################################
# END
###############################################################################

echo "=============================================="
echo "T2* PREPROCESSING COMPLETE"
echo "Subject: ${sub}"
echo "Session: ${ses}"
echo "=============================================="

