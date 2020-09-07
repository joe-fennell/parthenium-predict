import numpy as np
import os
import xarray as xr
import pickle

DEFAULT_MODEL = 'trained_classifier_2020-06-11T12_20_05.045413.pl'
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__),
                                  'models',
                                  DEFAULT_MODEL)


def predict_NetCDF(dataset, feature_descriptor, pipeline=None,
                   scl_include=[4, 5, 7], predict_proba=False,
                   predict_proba_class=1):
    """
    Maps an xarray S2 dataset to an sklearn-ready array and applied a pipeline.

    Parameters
    ----------
    dataset : xarray.Dataset
        NetCDF xarray object containing all bands needed for classifier
    feature_descriptor : dict
        key is a dimension in `dataset` and value is a list of coordinates
    pipeline : dask_ml.wrappers.ParallelPostFit
        trained pipeline instance containing classifer/model
    scl_include : list
        list of SCL codes to leave unmasked
    predict_proba : bool
        If True return array of probability for class predict_proba_class
    predict_proba_class : int
        class to generate probability for if predict_proba is True. Ignored
        otherwise.

    """
    def _make_mask(mask):
        _mask = mask.copy()
        _mask -= _mask
        for s in scl_include:
            _mask = _mask + mask.where(mask == s, 0)
        return _mask != 0

    if pipeline is None:
        pipeline = pickle.load(open(DEFAULT_MODEL_PATH,'rb'))

    # generate a subset of dates, bands, positions
    subset = dataset.sel(**feature_descriptor)
    feature_dims = set(feature_descriptor.keys())
    pix_dims = set(subset.dims).difference(feature_dims)
    # stack data into features and observations
    subset = subset.stack(obs=tuple(pix_dims), feats=tuple(feature_dims))
    mask = _make_mask(subset.SCL)
    reflectance = (subset.reflectance.astype(float)/10000).transpose(
        'obs', ...)
    # remove masked values from final array
    input_ = reflectance.where(mask).dropna('obs')
    # generate ouput array
    out = xr.DataArray(np.empty(reflectance.shape[0]) * np.nan,
                       coords={'obs': reflectance.obs},
                       dims='obs')
    if predict_proba:
        out[mask] = pipeline.predict_proba(input_)[:, predict_proba_class]
    else:
        out[mask] = pipeline.predict(input_)
    return out.unstack()


def generate_test_DataSet():
    """
    Generate test S2 data

    """
    shape = (12, 12, 10, 2)
    bands = ['B02', 'B03', 'B04', 'B05', 'B06',
             'B07', 'B08', 'B8A', 'B11', 'B12']
    reflectance = xr.DataArray(np.random.randint(1, 10000, shape),
                               dims=['y', 'x', 'band', 'date'],
                               coords={'y': np.arange(12),
                                       'x': np.arange(12),
                                       'band': bands}
                               )
    SCL = xr.DataArray(np.repeat(np.arange(12), 12*2).
                       reshape((12, 12, 2)),
                       dims=['y', 'x', 'date'],
                       coords={'y': np.arange(12),
                               'x': np.arange(12)})
    return xr.Dataset({'reflectance': reflectance, 'SCL': SCL})
