import numpy as np
import sys
sys.path.append(r'C:/Users/w37262do/Documents/git/IR_preprocessing')
sys.path.append('C:/Users/w37262do/Documents/git/PyIR/src')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing')
sys.path.append(r'/mnt/iusers01/eee01/w37262do/scratch/IR_preprocessing/PyIR/src')
from pyir_spectralcollection import PyIR_SpectralCollection as pir

def get_processing_tk_from_name(pipeline_name,):
    if pipeline_name == 'identity':
        return PreprocessingBase()
    if pipeline_name == 'QCL2026':
        return QCL2026()


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