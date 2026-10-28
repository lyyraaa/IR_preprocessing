import numpy as np
import sys

from openpyxl.descriptors import MinMax
from pybaselines import Baseline
sys.path.append(r'C:/Users/w37262do/Documents/git/IR_preprocessing')
sys.path.append('C:/Users/w37262do/Documents/git/PyIR/src')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing/PyIR/src')
from pyir_spectralcollection import PyIR_SpectralCollection as pir
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
import skimage.exposure


class PreprocessingBase():
    def __init__(self,args):
        self.args = args
        self.kit = pir()

    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = spectra
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        if mask is None:
            transformed_image, transformed_wavenumbers = self.spectral_transform(image.reshape(h*w,c),wavenumbers)
            return transformed_image.reshape(h, w, -1), transformed_wavenumbers
        else:
            transformed_spectra, transformed_wavenumbers = self.spectral_transform(image[mask==1],wavenumbers)
            transformed_image = np.zeros((h,w,transformed_spectra.shape[-1]))
            transformed_image[np.where(mask==1)] = transformed_spectra
            return transformed_image, transformed_wavenumbers

class MNFSmooth(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,_,_ = self.kit.imnf_denoise(
            spectra,
            wavenumbers=wavenumbers,
            bands=int(self.args[0]),
            noise_method=self.args[1],) # 'silent_region' for spatially-invariant, or 'traditional_MNF' for traditional MNF
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class QCL2026(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = self.kit.all_spec_min2zero(spectra)
        transformed_spectra = self.kit.vector_norm(transformed_spectra)
        transformed_spectra, transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, wavenumbers, 21, 5, 1)
        return transformed_spectra, transformed_wavenumbers

class PCASmooth(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        pca = PCA(n_components=int(self.args[0]))
        pca.fit(spectra)
        transformed_spectra = pca.inverse_transform(pca.transform(spectra))
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class modpoly(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        baseline_fitter = Baseline(x_data=wavenumbers)
        transformed_spectra = np.array([spec - baseline_fitter.modpoly(spec,poly_order=int(self.args[0]))[0] for spec in spectra])
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class asls(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        baseline_fitter = Baseline(x_data=wavenumbers)
        transformed_spectra = np.array([spec - baseline_fitter.asls(spec,lam=int(self.args[0]))[0] for spec in spectra])
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class rubberband(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        baseline_fitter = Baseline(x_data=wavenumbers)
        transformed_spectra = np.array(
            [spec - baseline_fitter.rubberband(spec,lam=float(self.args[0]))[0] for spec in spectra])
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class min2zero(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = self.kit.all_spec_min2zero(spectra)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class vector_norm(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = self.kit.vector_norm(spectra)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class feature_norm(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = self.kit.feature_norm(data=spectra,feature_pos=int(self.args[0]),wavenumbers=wavenumbers)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class remove_wax(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.remove_wax(spectra,wavenumbers)
        return transformed_spectra, transformed_wavenumbers

class baseline_correct(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,_ = self.kit.baseline_correct(spectra)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class baseline_correct_mean(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,_ = self.kit.baseline_correct(spectra, mean=True)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class sg_filter(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(
            spectra, wavenumbers, int(self.args[0]), int(self.args[1]), int(self.args[2])
        )
        return transformed_spectra, transformed_wavenumbers

class remove_glycogen(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.remove_range(1060,952, spectra, wavenumbers)
        return transformed_spectra, transformed_wavenumbers

class MinMax(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = spectra - spectra.min(1,keepdims=True)
        transformed_spectra = transformed_spectra / transformed_spectra.max(1,keepdims=True)
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

class Gaussian(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    # args = sigma, order
    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        transformed_image = ndimage.gaussian_filter(image, sigma=float(self.args[0]), order=int(self.args[1]), axes=(0,1))
        return transformed_image, wavenumbers

class GaussianLaplace(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        transformed_image = ndimage.gaussian_laplace(image, float(self.args[0]), axes=(0,1))
        return transformed_image, wavenumbers

class Median(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        transformed_image = ndimage.median_filter(image, size=(int(self.args[0]),int(self.args[0])), axes=(0,1))
        return transformed_image, wavenumbers

class SpatialRescale(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        mins = image[mask == 1].min(axis=0)
        maxs = image[mask == 1].max(axis=0)
        transformed_image = np.stack([
            skimage.exposure.rescale_intensity(image[..., x], in_range=(mins[x], maxs[x]), out_range=(0, 1)) for x in
            range(len(wavenumbers))
        ], axis=-1)
        return transformed_image, wavenumbers

class Hist(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape
        transformed_image = np.stack([
            skimage.exposure.equalize_hist(image[...,x],mask=mask) for x in range(len(wavenumbers))
        ],axis=-1)
        return transformed_image, wavenumbers

class DerivHist1(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra, transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 21, 5, 1)
        return transformed_spectra, transformed_wavenumbers

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape

        transformed_spectra, transformed_wavenumbers = self.spectral_transform(image[mask == 1], wavenumbers)
        transformed_image = np.zeros((h, w, transformed_spectra.shape[-1]))
        transformed_image[np.where(mask == 1)] = transformed_spectra

        transformed_image = np.stack([
            skimage.exposure.equalize_hist(transformed_image[..., x], mask=mask) for x in range(len(transformed_wavenumbers))
        ],axis=-1)
        return transformed_image, transformed_wavenumbers

class DerivHist2(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra, transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 21, 5, 2)
        return transformed_spectra, transformed_wavenumbers

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape

        transformed_spectra, transformed_wavenumbers = self.spectral_transform(image[mask == 1], wavenumbers)
        transformed_image = np.zeros((h, w, transformed_spectra.shape[-1]))
        transformed_image[np.where(mask == 1)] = transformed_spectra

        transformed_image = np.stack([
            skimage.exposure.equalize_hist(transformed_image[..., x], mask=mask) for x in range(len(transformed_wavenumbers))
        ],axis=-1)
        return transformed_image, transformed_wavenumbers

class AdaptiveHist(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        print("WARNING: This function should not be called in this spatial transform")

    def image_transform(self, image, wavenumbers=None, mask=None):
        h,w,c = image.shape

        image = np.clip(image, -1, 1)

        transformed_image = np.stack([
            skimage.exposure.equalize_adapthist(image[...,x]) for x in range(len(wavenumbers))
        ],axis=-1)
        return transformed_image, wavenumbers


class GreedyPipeline(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):

        transformed_spectra, transformed_wavenumbers = spectra, wavenumbers

        # DENOISE STEP
        if self.args[0] == 'sg':
            transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, transformed_wavenumbers, 21, 5, 0)
        elif self.args[0] == 'pca':
            pca = PCA(n_components=30)
            pca.fit(transformed_spectra)
            transformed_spectra = pca.inverse_transform(pca.transform(transformed_spectra))
        elif self.args[0] == 'mnf':
            transformed_spectra, _, _ = self.kit.imnf_denoise(transformed_spectra,wavenumbers=transformed_wavenumbers,bands=30,noise_method='traditional_MNF')  # 'silent_region' for spatially-invariant, or 'traditional_MNF' for traditional MNF
        elif self.args[0] == 'imnf':
            transformed_spectra, _, _ = self.kit.imnf_denoise(transformed_spectra,wavenumbers=transformed_wavenumbers,bands=30,noise_method='silent_region')

        # WAX REMOVAL STEP
        if self.args[1] == 'waxremove':
            transformed_spectra,transformed_wavenumbers = self.kit.remove_wax(transformed_spectra,transformed_wavenumbers)

        # BASELINE STEP
        baseline_fitter = Baseline(x_data=transformed_wavenumbers)
        if self.args[2] == 'min2zero':
            transformed_spectra = self.kit.all_spec_min2zero(transformed_spectra)
        elif self.args[2] == 'rubberband':
            transformed_spectra = np.array(
                [spec - baseline_fitter.rubberband(spec, lam=0)[0] for spec in transformed_spectra]
            )
        elif self.args[2] == 'poly':
            transformed_spectra = np.array(
                [spec - baseline_fitter.modpoly(spec, poly_order=2)[0] for spec in transformed_spectra]
            )
        elif self.args[2] == 'asls':
            transformed_spectra = np.array(
                [spec - baseline_fitter.asls(spec)[0] for spec in transformed_spectra]
            )

        # DERIVATIVE STEP
        if self.args[3] == '1st':
            transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, transformed_wavenumbers, 21, 5, 1)
        elif self.args[3] == '2nd':
            transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, transformed_wavenumbers, 21, 5, 2)
        elif self.args[3] == '1staus':
            transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, transformed_wavenumbers, 41, 2, 1)
        elif self.args[3] == '2ndaus':
            transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(transformed_spectra, transformed_wavenumbers, 41, 2, 2)

        # NORMALISE STEP
        if self.args[4] == 'minmax':
            transformed_spectra = transformed_spectra - transformed_spectra.min(1, keepdims=True)
            transformed_spectra = transformed_spectra / transformed_spectra.max(1, keepdims=True)
        elif self.args[4] == 'amidei':
            transformed_spectra = self.kit.feature_norm(data=transformed_spectra, feature_pos=1650,wavenumbers=transformed_wavenumbers)
        elif self.args[4] == 'amideii':
            transformed_spectra = self.kit.feature_norm(data=transformed_spectra, feature_pos=1550,wavenumbers=transformed_wavenumbers)
        elif self.args[4] == 'vectornorm':
            transformed_spectra = self.kit.vector_norm(transformed_spectra)

        return transformed_spectra, transformed_wavenumbers

class GreedyHist(GreedyPipeline):
    def image_transform(self, image, wavenumbers=None, mask=None):
        h, w, c = image.shape

        transformed_spectra, transformed_wavenumbers = self.spectral_transform(image[mask == 1], wavenumbers)
        transformed_image = np.zeros((h, w, transformed_spectra.shape[-1]))
        transformed_image[np.where(mask == 1)] = transformed_spectra

        transformed_image = np.stack([
            skimage.exposure.equalize_hist(transformed_image[..., x], mask=mask) for x in
            range(len(transformed_wavenumbers))
        ], axis=-1)
        return transformed_image, transformed_wavenumbers

class ConcatDeriv(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):

        transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 21, 5, 1)
        transformed_spectra = self.kit.vector_norm(transformed_spectra)

        transformed_spectra = np.concatenate((spectra, transformed_spectra), axis=-1)
        transformed_wavenumbers = np.concatenate((wavenumbers, transformed_wavenumbers), axis=-1)

        return transformed_spectra, transformed_wavenumbers

class GreedyOptimal(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):

        transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, int(self.args[0]), int(self.args[1]), 1)
        transformed_spectra = self.kit.vector_norm(transformed_spectra)

        return transformed_spectra, transformed_wavenumbers

trans_dict = {
    ### Baseline
    'identity':PreprocessingBase,
    'raw':PreprocessingBase,
    ### Individual transforms
    'min2zero':min2zero,
    'vector_norm':vector_norm,
    'feature_norm':feature_norm,
    'remove_wax':remove_wax,
    'baseline_correct':baseline_correct,
    'baseline_correct_mean':baseline_correct_mean,
    'sg_filter':sg_filter,
    'remove_glycogen':remove_glycogen,
    'pcasmooth':PCASmooth,
    'mnfsmooth':MNFSmooth,
    'minmax':MinMax,
    ### Established pipelines
    'QCL2026':QCL2026,
    ### New baseline ones
    'rubberband':rubberband,
    'modpoly':modpoly,
    'asls':asls,
    'greedy':GreedyPipeline,
    'greedy_optimal':GreedyOptimal,
    ### Spatial transforms
    'gaussian':Gaussian,
    'gol':GaussianLaplace,
    'median':Median,
    'rescale':SpatialRescale,
    'hist':Hist,
    'ahist':AdaptiveHist,
    'derivhist1':DerivHist1,
    'derivhist2':DerivHist2,
    'greedyhist':GreedyHist,
    # Concatenating transforms
    'concatderiv':ConcatDeriv,
}
def get_processing_tk_from_name(pipeline_name,args=None):
    return trans_dict[pipeline_name](args)