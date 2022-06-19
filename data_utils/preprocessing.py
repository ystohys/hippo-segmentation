import os
import warnings
import shutil
import subprocess
import collections
import SimpleITK as sitk
import nibabel as nib
import nilearn.image as niimage
import numpy as np
import matplotlib.pyplot as plt
from intensity_normalization.plot.histogram import HistogramPlotter

RAW_DATA_PATH = 'Round1_Processed_Data'


def n4bfc(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_raw.nii'):
                print('N4 Bias Field Correction running for: {0}'.format(xid), end='\r', flush=True)
                inputImage = sitk.ReadImage(os.path.join(data_path, xid, filename), sitk.sitkFloat32)
                image = inputImage

                maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
                #sitk.WriteImage(maskImage, os.path.join(data_path, xid, '{0}_mask.nii'.format(xid)))

                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected_image = corrector.Execute(image, maskImage)

                log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
                corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )

                sitk.WriteImage(corrected_image_full_resolution, os.path.join(data_path, xid, '{0}_bfc.nii'.format(xid)))

                print("Finished N4 Bias Field Correction for {0}".format(xid), end='\r', flush=True)


def brain_extraction(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_bfc.nii'): # Brain extraction after bias field correction
                print('Running brain extraction for {0}'.format(xid), end='\r', flush=True)
                subprocess.run(["robex", os.path.join(data_path, xid, filename),
                                "-os",
                                os.path.join(data_path, xid, '{}_stripped.nii'.format(xid)),
                                "-om",
                                os.path.join(data_path, xid, '{}_masked.nii'.format(xid))])
                
                
def ws_intensity_norm(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join('mri_data', 'wsnorm_stripped_imgs', xid), exist_ok=True)
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_stripped.nii'):
                print('WhiteStripe normalizing for {0}'.format(xid), end='\r', flush=True)
                subprocess.run(["ws-normalize", os.path.join(data_path, xid, filename),
                                "-o", 
                                os.path.join('mri_data', 'wsnorm_stripped_imgs', xid, '{}_wsnorm.nii'.format(xid))])


def zs_intensity_norm(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join('mri_data', 'znorm_stripped_imgs', xid), exist_ok=True)
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_stripped.nii'):
                print('Z-score normalizing for {0}'.format(xid), end='\r', flush=True)
                subprocess.run(["zscore-normalize", os.path.join(data_path, xid, filename),
                                "-o",
                                os.path.join('mri_data', 'znorm_stripped_imgs', xid, '{}_znorm.nii'.format(xid))])


def fcm_wm_intensity_norm(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join('mri_data', 'fcmnorm_wm_stripped_imgs', xid), exist_ok=True)
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_stripped.nii'):
                print('FCM intensity normalizing for {0}'.format(xid), end='\r', flush=True)
                subprocess.run(["fcm-normalize", os.path.join(data_path, xid, filename),
                                "-o",
                                os.path.join('mri_data', 'fcmnorm_wm_stripped_imgs', xid, '{}_fcmnorm.nii'.format(xid)),
                                "-tt", "wm"])


def fcm_gm_intensity_norm(data_path):
    for xid in os.listdir(data_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join('mri_data', 'fcmnorm_gm_stripped_imgs', xid), exist_ok=True)
        for filename in os.listdir(os.path.join(data_path, xid)):
            if filename.endswith('_stripped.nii'):
                print('FCM intensity normalizing for {0}'.format(xid), end='\r', flush=True)
                subprocess.run(["fcm-normalize", os.path.join(data_path, xid, filename),
                                "-o",
                                os.path.join('mri_data', 'fcmnorm_gm_stripped_imgs', xid, '{}_fcmnorm.nii'.format(xid)),
                                "-tt", "gm"])

                                
def port_over_masks(old_path, new_path):
    for xid in os.listdir(new_path):
        if xid.startswith('.'):
            continue
        for filename in os.listdir(os.path.join(old_path, xid)):
            if filename.endswith('_L.nii') and not filename.endswith('_CSF_L.nii'):
                shutil.copy(os.path.join(old_path, xid, filename), os.path.join(new_path, xid, ''))
                os.rename(os.path.join(new_path, xid, filename), 
                          os.path.join(new_path, xid, 'ADNI_{}_L.nii'.format(xid)))
                
            if filename.endswith('_R.nii') and not filename.endswith('_CSF_R.nii'):
                shutil.copy(os.path.join(old_path, xid, filename), os.path.join(new_path, xid, ''))
                os.rename(os.path.join(new_path, xid, filename), 
                          os.path.join(new_path, xid, 'ADNI_{}_R.nii'.format(xid)))
        
        for oth_filename in os.listdir(os.path.join(new_path, xid)):
            if oth_filename.endswith('gm_membership.nii'):
                os.remove(os.path.join(new_path, xid, oth_filename))
                

def minmax_scale(imgs_path, new_dir, uncropped_suffix=None, cropped=True):
    """
    WARNING: This is not the conventional min-max.
    """
    if cropped:
        for xid in os.listdir(imgs_path):
            if xid.startswith('.'):
                continue
            os.makedirs(os.path.join(new_dir, xid), exist_ok=True)
            for yid in os.listdir(os.path.join(imgs_path, xid)):
                if yid.startswith('.'):
                    continue
                elif yid.endswith('_LB.nii'):
                    tmp_nii = nib.load(os.path.join(imgs_path, xid, 'SMALL_{}_LB.nii'.format(xid)))
                    tmp_nii_arr = tmp_nii.get_fdata()
                    new_nii_arr = (tmp_nii_arr - tmp_nii_arr.min()) / (np.percentile(tmp_nii_arr, 99.9) - tmp_nii_arr.min())
                    new_nii_arr = (2 * new_nii_arr) - 1
                    new_nii = nib.Nifti1Image(new_nii_arr, tmp_nii.affine)
                    nib.save(new_nii, os.path.join(new_dir, xid, 'SMALL_{}_LB'.format(xid)))
                elif yid.endswith('_LH.nii'):
                    shutil.copy(os.path.join(imgs_path, xid, 'SMALL_{}_LH.nii'.format(xid)),
                                os.path.join(new_dir, xid, ''))
                elif yid.endswith('_RB.nii'):
                    tmp_nii = nib.load(os.path.join(imgs_path, xid, 'SMALL_{}_RB.nii'.format(xid)))
                    tmp_nii_arr = tmp_nii.get_fdata()
                    new_nii_arr = (tmp_nii_arr - tmp_nii_arr.min()) / (np.percentile(tmp_nii_arr, 99.9) - tmp_nii_arr.min())
                    new_nii_arr = (2 * new_nii_arr) - 1
                    new_nii = nib.Nifti1Image(new_nii_arr, tmp_nii.affine)
                    nib.save(new_nii, os.path.join(new_dir, xid, 'SMALL_{}_RB'.format(xid)))
                elif yid.endswith('_RH.nii'):
                    shutil.copy(os.path.join(imgs_path, xid, 'SMALL_{}_RH.nii'.format(xid)),
                                os.path.join(new_dir, xid, ''))
    else:
        for xid in os.listdir(imgs_path):
            if xid.startswith('.'):
                continue
            os.makedirs(os.path.join(new_dir, xid), exist_ok=True)
            for yid in os.listdir(os.path.join(imgs_path, xid)):
                if yid.startswith('.'):
                    continue
                elif yid.endswith(uncropped_suffix):
                    tmp_nii = nib.load(os.path.join(imgs_path, xid, '{0}_{1}'.format(xid, uncropped_suffix)))
                    tmp_nii_arr = tmp_nii.get_fdata()
                    new_nii_arr = (tmp_nii_arr-tmp_nii_arr.min()) / (np.percentile(tmp_nii_arr, 99.99)-tmp_nii_arr.min())
                    new_nii_arr = (2 * new_nii_arr) - 1
                    new_nii = nib.Nifti1Image(new_nii_arr, tmp_nii.affine)
                    nib.save(new_nii, os.path.join(new_dir, xid, '{0}_{1}'.format(xid, uncropped_suffix)))
                elif yid.endswith('_L.nii'):
                    shutil.copy(os.path.join(imgs_path, xid, 'ADNI_{}_L.nii'.format(xid)),
                                os.path.join(new_dir, xid, ''))
                elif yid.endswith('_R.nii'):
                    shutil.copy(os.path.join(imgs_path, xid, 'ADNI_{}_R.nii'.format(xid)),
                                os.path.join(new_dir, xid, ''))
                elif yid.endswith('_masked.nii'):
                    shutil.copy(os.path.join(imgs_path, xid, '{}_masked.nii'.format(xid)),
                                os.path.join(new_dir, xid, ''))


def zscore_standardise(imgs_path, new_dir):
    if 'cropped' not in imgs_path:
        warnings.warn(
            ("Your file path does not indicate that the images are cropped. "
             "This function should only be used on cropped images!"))
    for xid in os.listdir(imgs_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join(new_dir, xid), exist_ok=True)
        for yid in os.listdir(os.path.join(imgs_path, xid)):
            if yid.startswith('.'):
                continue
            elif yid.endswith('_LB.nii'):
                tmp_nii = nib.load(os.path.join(imgs_path, xid, 'SMALL_{}_LB.nii'.format(xid)))
                tmp_nii_arr = tmp_nii.get_fdata()
                new_nii_arr = (tmp_nii_arr - tmp_nii_arr.mean()) / (tmp_nii_arr.std())
                new_nii = nib.Nifti1Image(new_nii_arr, tmp_nii.affine)
                nib.save(new_nii, os.path.join(new_dir, xid, 'SMALL_{}_LB'.format(xid)))
            elif yid.endswith('_LH.nii'):
                shutil.copy(os.path.join(imgs_path, xid, 'SMALL_{}_LH.nii'.format(xid)),
                            os.path.join(new_dir, xid, ''))
            elif yid.endswith('_RB.nii'):
                tmp_nii = nib.load(os.path.join(imgs_path, xid, 'SMALL_{}_RB.nii'.format(xid)))
                tmp_nii_arr = tmp_nii.get_fdata()
                new_nii_arr = (tmp_nii_arr - tmp_nii_arr.mean()) / (tmp_nii_arr.std())
                new_nii = nib.Nifti1Image(new_nii_arr, tmp_nii.affine)
                nib.save(new_nii, os.path.join(new_dir, xid, 'SMALL_{}_RB'.format(xid)))
            elif yid.endswith('_RH.nii'):
                shutil.copy(os.path.join(imgs_path, xid, 'SMALL_{}_RH.nii'.format(xid)),
                            os.path.join(new_dir, xid, ''))


def crop_imgs_and_masks(imgs_path):
    for xid in os.listdir(imgs_path):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join('final_data', xid), exist_ok=True)
        curr_img = nib.load(os.path.join(imgs_path, xid, '{0}_wsnorm.nii'.format(xid)))
        cropped = niimage.crop_img(curr_img, rtol=1e-08, copy=True, pad=True, return_offset=False)
        nib.save(cropped, os.path.join('final_data', xid, '{}_cropped.nii'.format(xid)))
        for yid in os.listdir(os.path.join(imgs_path, xid)):
            if yid.endswith('CSF_L.nii'):
                old_mask = nib.load(os.path.join(imgs_path, xid, yid))
                new_mask = niimage.resample_img(old_mask, target_affine=cropped.affine, target_shape=cropped.shape)
                nib.save(new_mask, os.path.join('final_data', xid, '{}_CSF_L_cropped.nii'.format(xid)))
            elif yid.endswith('CSF_R.nii'):
                old_mask = nib.load(os.path.join(imgs_path, xid, yid))
                new_mask = niimage.resample_img(old_mask, target_affine=cropped.affine, target_shape=cropped.shape)
                nib.save(new_mask, os.path.join('final_data', xid, '{}_CSF_R_cropped.nii'.format(xid)))
            elif yid.endswith('_L.nii') and not yid.endswith('CSF_L.nii'):
                old_mask = nib.load(os.path.join(imgs_path, xid, yid))
                new_mask = niimage.resample_img(old_mask, target_affine=cropped.affine, target_shape=cropped.shape)
                nib.save(new_mask, os.path.join('final_data', xid, '{}_L_cropped.nii'.format(xid)))
            elif yid.endswith('_R.nii') and not yid.endswith('CSF_R.nii'):
                old_mask = nib.load(os.path.join(imgs_path, xid, yid))
                new_mask = niimage.resample_img(old_mask, target_affine=cropped.affine, target_shape=cropped.shape)
                nib.save(new_mask, os.path.join('final_data', xid, '{}_R_cropped.nii'.format(xid)))
                

def combine_lr_hcsf(imgs_path, cropped=False):
    for xid in os.listdir(imgs_path):
        if xid.startswith('.'):
            continue
        if cropped:
            if os.path.exists(os.path.join(imgs_path, xid, '{}_CSF_L_cropped.nii'.format(xid))) and os.path.exists(os.path.join(imgs_path, xid, '{}_CSF_R_cropped.nii'.format(xid))):
                csf_l = nib.load(os.path.join(imgs_path, xid, '{}_CSF_L_cropped.nii'.format(xid)))
                csf_r = nib.load(os.path.join(imgs_path, xid, '{}_CSF_R_cropped.nii'.format(xid)))
                csf_arr = csf_l.get_fdata() + csf_r.get_fdata()
                comb_csf = nib.Nifti1Image(csf_arr, csf_l.affine)
                nib.save(comb_csf, os.path.join(imgs_path, xid, '{}_CSF_LR_cropped.nii'.format(xid)))
            
            hipp_l = nib.load(os.path.join(imgs_path, xid, '{}_L_cropped.nii'.format(xid)))
            hipp_r = nib.load(os.path.join(imgs_path, xid, '{}_R_cropped.nii'.format(xid)))
            hipp_arr = hipp_l.get_fdata() + hipp_r.get_fdata()
            comb_hipp = nib.Nifti1Image(hipp_arr, hipp_l.affine)
            nib.save(comb_hipp, os.path.join(imgs_path, xid, '{}_HLR_cropped.nii'.format(xid)))
            
        if not cropped:
            if os.path.exists(os.path.join(imgs_path, xid, 'ADNI_{}_CSF_L.nii'.format(xid))) and os.path.exists(os.path.join(imgs_path, xid, 'ADNI_{}_CSF_R.nii'.format(xid))):
                csf_l = nib.load(os.path.join(imgs_path, xid, 'ADNI_{}_CSF_L.nii'.format(xid)))
                csf_r = nib.load(os.path.join(imgs_path, xid, 'ADNI_{}_CSF_R.nii'.format(xid)))
                csf_arr = csf_l.get_fdata() + csf_r.get_fdata()
                comb_csf = nib.Nifti1Image(csf_arr, csf_l.affine)
                nib.save(comb_csf, os.path.join(imgs_path, xid, 'ADNI_{}_CSF_LR.nii'.format(xid)))
            
            hipp_l = nib.load(os.path.join(imgs_path, xid, 'ADNI_{}_L.nii'.format(xid)))
            hipp_r = nib.load(os.path.join(imgs_path, xid, 'ADNI_{}_R.nii'.format(xid)))
            hipp_arr = hipp_l.get_fdata() + hipp_r.get_fdata()
            comb_hipp = nib.Nifti1Image(hipp_arr, hipp_l.affine)
            nib.save(comb_hipp, os.path.join(imgs_path, xid, 'ADNI_{}_HLR.nii'.format(xid)))


def plot_intensity_dist(imgs_path, img_suffix, title, cropped=False, mask_suffix='_masked.nii'):
    if cropped:
        imgs_arr = collections.deque([])
        for xid in os.listdir(imgs_path):
            if xid.startswith('.'):
                continue
            imgs_arr.append(nib.load(os.path.join(imgs_path, xid, 'SMALL_{0}_{1}'.format(xid, img_suffix))).get_fdata())
        hp = HistogramPlotter(title=title)
        _ = hp(imgs_arr, None)
        plt.show()
    else:
        imgs_arr = collections.deque([])
        masks_arr = collections.deque([])
        for xid in os.listdir(imgs_path):
            if xid.startswith('.'):
                continue
            imgs_arr.append(nib.load(os.path.join(imgs_path, xid, '{0}_{1}'.format(xid, img_suffix))).get_fdata())
            masks_arr.append(nib.load(os.path.join(imgs_path, xid, '{0}_{1}'.format(xid, mask_suffix))).get_fdata())

        hp = HistogramPlotter(title=title)
        _ = hp(imgs_arr, masks_arr)
        plt.show()


def plot_extreme_intensities(imgs_path, img_suffix, upper_bound, title):  # Use only for cropped images
    # imgs_arr = collections.deque([])
    plt.title(title)
    plt.xlabel('Voxel index (flattened)')
    plt.ylabel('Intensity')
    for xid in os.listdir(imgs_path):
        if xid.startswith('.'):
            continue
        y = nib.load(os.path.join(imgs_path, xid, 'SMALL_{0}_{1}'.format(xid, img_suffix))).get_fdata().flatten()
        x = (y > upper_bound).nonzero()
        y = y[x]
        plt.scatter(x, y)
    plt.show()


def find_max_boundary(path_to_imgs):
    boundaries_left = [[9999, 0],
                       [9999, 0],
                       [9999, 0]]
    boundaries_right = [[9999, 0],
                        [9999, 0],
                        [9999, 0]]
    for xid in os.listdir(path_to_imgs):
        if xid.startswith('.'):
            continue
        left_hip = nib.load(os.path.join(path_to_imgs, xid, 'ADNI_{}_L.nii'.format(xid)))
        right_hip = nib.load(os.path.join(path_to_imgs, xid, 'ADNI_{}_R.nii'.format(xid)))
        left_arr = left_hip.get_fdata()
        right_arr = right_hip.get_fdata()

        flag = False

        for i in range(left_arr.shape[0]):
            if not flag:
                if np.flatnonzero(left_arr[i,:,:]).size > 0:
                    flag = True
                    if i < boundaries_left[0][0]:
                        boundaries_left[0][0] = i
            elif flag:
                if np.flatnonzero(left_arr[i,:,:]).size == 0:
                    flag = False
                    if i > boundaries_left[0][1]:
                        boundaries_left[0][1] = i-1
                    break

        for ii in range(right_arr.shape[0]):
            if not flag:
                if np.flatnonzero(right_arr[ii,:,:]).size > 0:
                    flag = True
                    if ii < boundaries_right[0][0]:
                        boundaries_right[0][0] = ii
            elif flag:
                if np.flatnonzero(right_arr[ii,:,:]).size == 0:
                    flag = False
                    if ii > boundaries_right[0][1]:
                        boundaries_right[0][1] = ii-1
                    break

        for j in range(left_arr.shape[1]):
            if not flag:
                if np.flatnonzero(left_arr[:,j,:]).size > 0:
                    flag = True
                    if j < boundaries_left[1][0]:
                        boundaries_left[1][0] = j
            elif flag:
                if np.flatnonzero(left_arr[:,j,:]).size == 0:
                    flag = False
                    if j > boundaries_left[1][1]:
                        boundaries_left[1][1] = j-1
                    break

        for jj in range(right_arr.shape[1]):
            if not flag:
                if np.flatnonzero(right_arr[:,jj,:]).size > 0:
                    flag = True
                    if jj < boundaries_right[1][0]:
                        boundaries_right[1][0] = jj
            elif flag:
                if np.flatnonzero(right_arr[:,jj,:]).size == 0:
                    flag = False
                    if jj > boundaries_right[1][1]:
                        boundaries_right[1][1] = jj-1
                    break

        for k in range(left_arr.shape[2]):
            if not flag:
                if np.flatnonzero(left_arr[:,:,k]).size > 0:
                    flag = True
                    if k < boundaries_left[2][0]:
                        boundaries_left[2][0] = k
            elif flag:
                if np.flatnonzero(left_arr[:,:,k]).size == 0:
                    flag = False
                    if k > boundaries_left[2][1]:
                        boundaries_left[2][1] = k-1
                    break

        for kk in range(right_arr.shape[2]):
            if not flag:
                if np.flatnonzero(right_arr[:,:,kk]).size > 0:
                    flag = True
                    if kk < boundaries_right[2][0]:
                        boundaries_right[2][0] = kk
            elif flag:
                if np.flatnonzero(right_arr[:,:,kk]).size == 0:
                    flag = False
                    if kk > boundaries_right[2][1]:
                        boundaries_right[2][1] = kk-1
                    break

    return boundaries_left, boundaries_right


def get_bounding_boxes(left_bound,
                       right_bound,
                       path_to_imgs,
                       brain_file_suffix,
                       new_path):
    for xid in os.listdir(path_to_imgs):
        if xid.startswith('.'):
            continue
        os.makedirs(os.path.join(new_path, xid), exist_ok=True)
        left_mask = nib.load(os.path.join(path_to_imgs, xid, 'ADNI_{}_L.nii'.format(xid)))
        right_mask = nib.load(os.path.join(path_to_imgs, xid, 'ADNI_{}_R.nii'.format(xid)))
        whole_brain = nib.load(os.path.join(path_to_imgs, xid, '{}_{}'.format(xid, brain_file_suffix)))

        left_arr = left_mask.get_fdata()
        right_arr = right_mask.get_fdata()
        brain_arr = whole_brain.get_fdata()

        new_left_arr = left_arr[left_bound[0][0]:left_bound[0][1],
                                left_bound[1][0]:left_bound[1][1],
                                left_bound[2][0]:left_bound[2][1]]
        new_right_arr = right_arr[right_bound[0][0]:right_bound[0][1],
                                  right_bound[1][0]:right_bound[1][1],
                                  right_bound[2][0]:right_bound[2][1]]
        left_brain_arr = brain_arr[left_bound[0][0]:left_bound[0][1],
                                   left_bound[1][0]:left_bound[1][1],
                                   left_bound[2][0]:left_bound[2][1]]
        right_brain_arr = brain_arr[right_bound[0][0]:right_bound[0][1],
                                    right_bound[1][0]:right_bound[1][1],
                                    right_bound[2][0]:right_bound[2][1]]
        new_left = nib.Nifti1Image(new_left_arr, left_mask.affine)
        new_right = nib.Nifti1Image(new_right_arr, right_mask.affine)
        left_brain = nib.Nifti1Image(left_brain_arr, whole_brain.affine)
        right_brain = nib.Nifti1Image(right_brain_arr, whole_brain.affine)
        nib.save(new_left, os.path.join(new_path, xid, 'SMALL_{}_LH.nii'.format(xid)))
        nib.save(new_right, os.path.join(new_path, xid, 'SMALL_{}_RH.nii'.format(xid)))
        nib.save(left_brain, os.path.join(new_path, xid, 'SMALL_{}_LB.nii'.format(xid)))
        nib.save(right_brain, os.path.join(new_path, xid, 'SMALL_{}_RB.nii'.format(xid)))
    print('DONE!')

        
if __name__ == "__main__":
    n4bfc('mri_data/unprocessed_data') # Bias field correction
    brain_extraction('mri_data/unprocessed_data') # Brain extraction (skullstripping)
    ws_intensity_norm('mri_data/unprocessed_data') # Intensity Normalization (WhiteStripe normalization)
    fcm_gm_intensity_norm('mri_data/unprocessed_data') # FCM Norm
    port_over_masks('mri_data/unprocessed_data', 'mri_data/wsnorm_stripped_imgs')
    port_over_masks('mri_data/unprocessed_data', 'mri_data/fcmnorm_stripped_imgs')
