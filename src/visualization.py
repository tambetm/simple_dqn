from neon.data import DataIterator
from neon.callbacks.callbacks import DeconvCallback
from nvis.figure import deconv_summary_page
from nvis.data import h5_deconv_data
import h5py

def visualize(model, data, max_fm, filename):
  data_shape = data.shape
  data = data.reshape((data.shape[0], -1))
  dataset = DataIterator(data, lshape=data_shape[1:])

  deconv_file = h5py.File("no_file", driver='core', backing_store=False)
  deconv = DeconvCallback(deconv_file, model, dataset, dataset,
      max_fm=max_fm, dataset_pct=100)
  deconv.on_train_end()

  deconv_data = h5_deconv_data(deconv_file)
  deconv_summary_page(filename, deconv_data, max_fm)
