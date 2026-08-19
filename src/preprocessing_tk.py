import numpy as np
import sys
from pybaselines import Baseline
sys.path.append(r'C:/Users/w37262do/Documents/git/IR_preprocessing')
sys.path.append('C:/Users/w37262do/Documents/git/PyIR/src')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing/PyIR/src')
from pyir_spectralcollection import PyIR_SpectralCollection as pir
from sklearn.decomposition import PCA


class PreprocessingBase():
    def __init__(self,args):
        self.args = args
        self.kit = pir()

    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra = spectra
        transformed_wavenumbers = wavenumbers
        return transformed_spectra, transformed_wavenumbers

    def image_transform(self, image, wavenumbers=None):
        h,w,c = image.shape
        transformed_image, transformed_wavenumbers = self.spectral_transform(image.reshape(h*w,c),wavenumbers)
        return transformed_image.reshape(h,w,-1), transformed_wavenumbers


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
        transformed_spectra = self.kit.feature_norm(spectra)
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
            spectra, wavenumbers, int(self.args[0]), int(self.args[0]), int(self.args[0])
        )
        return transformed_spectra, transformed_wavenumbers

class remove_glycogen(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.remove_range(1060,952, spectra, wavenumbers)
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
    ### Established pipelines
    'QCL2026':QCL2026,
    ### New baseline ones
    'rubberband':rubberband,
    'modpoly':modpoly,
    'pcasmooth':PCASmooth,
    # todo: MNF smoothing
    # todo: fix feature norm
    # todo: channel standardisation?
    # todo: ALS baseline

}
def get_processing_tk_from_name(pipeline_name,args=None):
    return trans_dict[pipeline_name](args)
