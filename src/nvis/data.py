# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import h5py
import numpy as np


def convert_rgb_to_bokehrgba(img_data, downsample=1):
    """
    Convert RGB image to two-dimensional array of RGBA values (encoded as 32-bit integers)
    (required by Bokeh). The functionality is currently not available in Bokeh.
    An issue was raised here: https://github.com/bokeh/bokeh/issues/1699 and this function is a
    modified version of the suggested solution.

    Arguments:
        img_data: img (ndarray, shape: [N, M, 3], dtype: uint8): image data
        dh: height of image
        dw: width of image

    Returns:
        img (ndarray): 2D image array of RGBA values
    """
    if img_data.dtype != np.uint8:
        raise NotImplementedError

    if img_data.ndim != 3:
        raise NotImplementedError

    # downsample for render performance, v-flip since plot origin is bottom left
    # img_data = np.transpose(img_data, (1,2,0))
    img_data = img_data[::-downsample, ::downsample, :]
    img_h, img_w, C = img_data.shape

    # add an alpha channel to the image and recast from pixels of u8u8u8u8 to u32
    #bokeh_img = np.dstack([img_data, 255 * np.ones((img_h, img_w), np.uint8)])
    #final_image = bokeh_img.reshape(img_h, img_w * (C+1)).view(np.uint32)
    # put last 3 frames into separate color channels and add alpha channel
    bokeh_img = np.dstack([img_data[:,:,1], img_data[:,:,2], img_data[:,:,3], 255 * np.ones((img_h, img_w), np.uint8)])
    final_image = bokeh_img.reshape(img_h, img_w * 4).view(np.uint32)

    return final_image


def h5_deconv_data(f):
    """
    Read deconv visualization data from hdf5 file.

    Returns:
        list of lists. Each inner list represents one layer, and consists of
        tuples (fm, deconv_data)
    """
    ret = list()

    if 'deconv' not in f.keys():
        return None
    act_data = f['deconv/max_act']
    img_data = f['deconv/img']

    for layer in act_data.keys():
        layer_data = list()
        for fm in range(act_data[layer]['vis'].shape[0]):

            # to avoid storing entire dataset, imgs are cached as needed, have to look up
            batch_ind, img_ind = act_data[layer]['batch_img'][fm]
            img_store = img_data['batch_{}'.format(batch_ind)]
            img_cache_ofs = img_store.attrs[str(img_ind)]

            # have to convert from rgb to rgba and cast as uint32 dtype for bokeh
            plot_img = convert_rgb_to_bokehrgba(img_store['HWC_uint8'][:, :, :, img_cache_ofs])
            plot_deconv = convert_rgb_to_bokehrgba(act_data[layer]['vis'][fm])

            layer_data.append((fm, plot_deconv, plot_img))

        ret.append((layer, layer_data))

    return ret
