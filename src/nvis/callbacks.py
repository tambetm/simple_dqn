# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import logging
import time
import numpy as np
import sys

from neon.callbacks.callbacks import Callback
from neon.layers import Convolution

logger = logging.getLogger(__name__)


class DeconvCallback(Callback):
    """
    Callback to store data after projecting activations back to pixel space using
    guided backpropagation.  See [Springenberg2014]_ for details.  Meant to be
    used for visualization purposes via nvis.

    Arguments:
        train_set (NervanaDataIterator): the training dataset
        max_fm (int, optional): Maximum number of feature maps to visualize per
                                layer.  Defaults to 16.
        dataset_pct (float, optional): Initial portion of validation dataset to
                                       use in finding maximum activations.
                                       Defaults to 25.0 (25%).

    Notes:

    .. [Springenberg2014] http://arxiv.org/abs/1412.6806
    """
    def __init__(self, train_set, valid_set, max_fm=16, dataset_pct=25):
        super(DeconvCallback, self).__init__(epoch_freq=1)
        self.train_set = train_set
        self.valid_set = valid_set
        self.max_fm = max_fm
        self.dataset_pct = dataset_pct
        self.name = "Guided Bprop"

    def _progress_update(self, tag, curr, total, unit, time, blockchar=u'\u2588'):
        # clear and redraw progress bar
        max_bar_width = 20
        bar_width = int(float(curr) / total * max_bar_width)
        s = u'Visualization  [{} |{:<%s}| {:4}/{:<4} {}, {:.2f}s]' % max_bar_width
        progress_string = s.format(tag, blockchar * bar_width, curr, total, unit, time)
        sys.stdout.write('\r' + progress_string.encode('utf-8'))
        sys.stdout.flush()

    def on_train_end(self, callback_data, model):
        # TODO: generalize for more complex topologies
        layers = model.layers.layers
        self.raw_img_cache = dict()
        self.raw_img_key = dict()
        C, H, W = layers[0].in_shape
        msg = "{} Visualization of {} feature maps per layer:"
        logger.info(msg.format(self.name, self.max_fm))

        for l, lyr in enumerate(layers):
            if isinstance(lyr, Convolution):
                K = lyr.convparams['K']
                num_fm = min(K, self.max_fm)

                lyr_data = callback_data.create_group("deconv/max_act/{0:04}".format(l))
                lyr_data.create_dataset("batch_img", (num_fm, 2), dtype='uint16')
                lyr_data.create_dataset("fm_loc", (num_fm, 1), dtype='int16')
                lyr_data.create_dataset("vis", (num_fm, H, W, C), dtype='uint8')
                lyr_data.create_dataset("activation", (num_fm, 1), dtype='float32')
                lyr_data['activation'][:] = -float('Inf')

        self.valid_set.reset()
        t_start = time.time()
        num_sampled_batches = int(self.dataset_pct / 100. *
                                  self.valid_set.nbatches + 0.5)
        for batch_ind, (x, t) in enumerate(self.valid_set, 0):

            if batch_ind > num_sampled_batches:
                break

            imgs_to_store = self.get_layer_acts(callback_data, model, x, batch_ind)

            self.store_images(callback_data, batch_ind, imgs_to_store, x, C, H, W)

            self._progress_update("Find Max Act Imgs", batch_ind,
                                  num_sampled_batches, "batches",
                                  time.time() - t_start)

        sys.stdout.write("\n")

        # Loop over every layer to visualize
        t_start = time.time()
        for i in range(1, len(layers) + 1):
            layer_ind = len(layers) - i

            if isinstance(layers[layer_ind], Convolution):
                num_fm, act_h, act_w = layers[layer_ind].out_shape
                act_size = act_h * act_w
                self.visualize_layer(callback_data, model, num_fm, act_size, layer_ind)
            self._progress_update("Compute " + self.name, i,
                                  len(layers), "layers",
                                  time.time() - t_start)

        sys.stdout.write("\n")

    def scale_to_rgb(self, img):
        """
        Convert float data to valid RGB values in the range [0, 255]

        Arguments:
            img (ndarray): the image data

        Returns:
            img (ndarray): image array with valid RGB values
        """
        img_min = np.min(img)
        img_rng = np.max(img) - img_min
        img_255 = img - img_min
        if img_rng > 0:
            img_255 /= img_rng
            img_255 *= 255.
        return img_255

    def store_images(self, callback_data, batch_ind, imgs_to_store, img_batch_data, C, H, W):
        n_imgs = len(imgs_to_store)
        if n_imgs:
            img_data = img_batch_data[:, imgs_to_store].get()
            img_store = callback_data.create_group('deconv/img/batch_'+str(batch_ind))

            # Store uint8 HWC formatted data for plotting
            img_hwc8 = img_store.create_dataset("HWC_uint8", (H, W, C, n_imgs),
                                                dtype='uint8', compression=True)
            img_hwc_f32 = np.transpose(img_data.reshape((C, H, W, n_imgs)), (1, 2, 0, 3))
            img_hwc8[:] = self.scale_to_rgb(img_hwc_f32)

            # keep image in native format to use for fprop in visualization
            # don't need this beyond runtime so avoid writing to file
            self.raw_img_cache[batch_ind] = img_data

            # Keep a lookup from img_ind -> file position
            # In order to store only needed imgs from batch in flat prealloc array
            self.raw_img_key[batch_ind] = dict()
            for i, img_idx in enumerate(imgs_to_store):
                img_store.attrs[str(img_idx)] = i
                self.raw_img_key[batch_ind][img_idx] = i

    def get_layer_acts(self, callback_data, model, x, batch_ind):
        imgs_to_store = set()

        for l, lyr in enumerate(model.layers.layers, 0):
            x = lyr.fprop(x, inference=True)

            if not isinstance(lyr, Convolution):
                continue

            num_fm, H, W = lyr.out_shape
            fm_argmax = self.be.zeros((num_fm, 1), dtype=np.int32)
            maxact_idx = self.be.array(np.arange(num_fm) * H * W * self.be.bsz, dtype=np.int32)

            act_data = callback_data["deconv/max_act/{0:04}".format(l)]

            all_acts = lyr.outputs.reshape((num_fm, H * W * self.be.bsz))
            all_acts_flat = lyr.outputs.reshape((num_fm * H * W * self.be.bsz))

            fm_argmax[:] = self.be.argmax(all_acts, axis=1)
            maxact_idx[:] = maxact_idx + fm_argmax
            acts_host = all_acts_flat[maxact_idx].get()
            fm_argmax_host = fm_argmax.get()

            num_fm_vis = min(num_fm, self.max_fm)
            for fm in range(num_fm_vis):

                argmax = fm_argmax_host[fm]
                img_ind = int(argmax % self.be.bsz)
                curr_max_act = acts_host[fm]

                if curr_max_act > act_data['activation'][fm]:
                    act_data['activation'][fm] = curr_max_act
                    act_data['batch_img'][fm] = batch_ind, img_ind
                    act_data['fm_loc'][fm] = argmax / self.be.bsz
                    imgs_to_store.add(img_ind)

        return list(imgs_to_store)

    def visualize_layer(self, callback_data, model, num_fm, act_size, layer_ind):
        be = model.be
        act_data = callback_data["deconv/max_act/{0:04}".format(layer_ind)]
        layers = model.layers.layers

        # Loop to visualize every feature map
        num_fm_vis = min(num_fm, self.max_fm)
        for fm in range(num_fm_vis):
            batch_ind, img_ind = act_data['batch_img'][fm]

            # Prepare a fake minibatch with just the max activation image for this fm
            img_batch = np.zeros((self.raw_img_cache[batch_ind].shape[0], be.bsz))
            img_cache_offs = self.raw_img_key[batch_ind][img_ind]
            img_batch[:, 0] = self.raw_img_cache[batch_ind][:, img_cache_offs]
            img_batch = be.array(img_batch)

            # Prep model internal state by fprop-ing img
            model.fprop(img_batch, inference=True)

            # Set the max activation at the correct feature map location
            fm_loc = act_data['fm_loc'][fm]
            activation = np.zeros((num_fm, act_size, be.bsz))
            activation[fm, fm_loc, :] = float(act_data['activation'][fm])
            activation = activation.reshape((num_fm * act_size, be.bsz))
            activation = be.array(activation)

            # Loop over the previous layers to perform deconv
            for i, l in enumerate(layers[layer_ind::-1], 0):
                if isinstance(l, Convolution):

                    # zero out w.r.t. current layer activations
                    activation[:] = be.maximum(activation, 0)

                    # output shape of deconv is the input shape of conv
                    C, H, W = [l.convparams[x] for x in ['C', 'H', 'W']]
                    out = be.empty((C * H * W, be.bsz))
                    l.be.bprop_conv(layer=l.nglayer, F=l.W, E=activation, grad_I=out)
                    activation = out

                    # zero out w.r.t to input from lower layer
                    layer_below_acts = layers[layer_ind - i].inputs
                    layer_below_acts[:] = be.greater(layer_below_acts, 0)
                    activation[:] = be.multiply(layer_below_acts, activation)

            C, H, W = layers[0].in_shape
            activation = activation.get().reshape((C, H, W, be.bsz))
            activation = np.transpose(activation, (1, 2, 0, 3))
            act_data['vis'][fm] = self.scale_to_rgb(activation[:, :, :, 0])
