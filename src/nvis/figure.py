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

# Imports should not be a requirement for building documentation
try:
    from bokeh.plotting import figure
    from bokeh.palettes import brewer
    from bokeh.models import Range1d
    from bokeh.embed import components
    from jinja2 import Template
except ImportError:
    pass


def image_fig(data, h, w, x_range, y_range, plot_size):
    fig = figure(x_range=x_range, y_range=y_range,
                 plot_width=plot_size, plot_height=plot_size,
                 toolbar_location=None)
    fig.image_rgba([data], x=[0], y=[0], dw=[w], dh=[h])
    fig.axis.visible = None
    fig.min_border = 0
    return fig


def deconv_figs(layer_name, layer_data, fm_max=8, plot_size=120):
    vis_keys = dict()
    img_keys = dict()
    fig_dict = dict()

    for fm_num, (fm_name, deconv_data, img_data) in enumerate(layer_data):

        if fm_num >= fm_max:
            break

        img_h, img_w = img_data.shape
        x_range = Range1d(start=0, end=img_w)
        y_range = Range1d(start=0, end=img_h)
        img_fig = image_fig(img_data, img_h, img_w, x_range, y_range, plot_size)
        deconv_fig = image_fig(deconv_data, img_h, img_w, x_range, y_range, plot_size)

        title = "{}_fmap_{:04d}".format(layer_name, fm_num)
        vis_keys[fm_num] = "vis_"+title
        img_keys[fm_num] = "img_"+title

        fig_dict[vis_keys[fm_num]] = deconv_fig
        fig_dict[img_keys[fm_num]] = img_fig

    return vis_keys, img_keys, fig_dict


def deconv_summary_page(filename, deconv_data, fm_max):
    fig_dict = dict()
    vis_keys = dict()
    img_keys = dict()
    for layer, layer_data in deconv_data:
        lyr_vis_keys, lyr_img_keys, lyr_fig_dict = deconv_figs(layer, layer_data, fm_max=fm_max)
        vis_keys[layer] = lyr_vis_keys
        img_keys[layer] = lyr_img_keys
        fig_dict.update(lyr_fig_dict)

    script, div = components(fig_dict)

    template = Template('''
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{{page_title}}</title>
        <style> div{float: left;} </style>
        <link rel="stylesheet"
              href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
              type="text/css" />
        <script type="text/javascript"
                src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
        {{ script }}
    </head>
    <body>

    {% for layer in sorted_layers %}
        <div id=Outer{{layer}} style="padding:20px">
        <div id={{layer}} style="background-color: #C6FFF1; padding:10px">
        Layer {{layer}}<br>
        {% for fm in vis_keys[layer].keys() %}
            <div id={{fm}} style="padding:10px">
            Feature Map {{fm}}<br>
            {{ div[vis_keys[layer][fm]] }}
            {{ div[img_keys[layer][fm]] }}
            </div>
        {% endfor %}
        </div>
        </div>

        <br><br>
    {% endfor %}
    </body>
</html>
''')

    with open(filename, 'w') as htmlfile:
        htmlfile.write(template.render(page_title="Deconv Visualization", script=script,
                                       div=div, vis_keys=vis_keys,
                                       img_keys=img_keys,
                                       sorted_layers=sorted(vis_keys)))
