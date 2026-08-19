import numpy as np
import sys
sys.path.append(r'C:/Users/w37262do/Documents/git/IR_preprocessing')
sys.path.append('C:/Users/w37262do/Documents/git/PyIR/src')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing/PyIR/src')
from pyir_spectralcollection import PyIR_SpectralCollection as pir



class PreprocessingBase():
    def __init__(self):
        self.name = "identity"
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

class deriv_1(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 21, 5, 1)
        return transformed_spectra, transformed_wavenumbers

class deriv_2(PreprocessingBase):
    def spectral_transform(self, spectra, wavenumbers=None):
        transformed_spectra,transformed_wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 21, 5, 2)
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
    'deriv_1':deriv_1,
    'deriv_2':deriv_2,
    'remove_glycogen':remove_glycogen,
    ### Established pipelines
    'QCL2026':QCL2026,
}
def get_processing_tk_from_name(pipeline_name,):
    return trans_dict[pipeline_name]()
