## For example usage see:
## https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb

# include tf slim models repo (tf_models)
import sys
import os
sys.path.append("../tf_models/research/slim")

import tensorflow.compat.v1 as tf
import tf_slim as slim
from nets.mobilenet import mobilenet_v2
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
from nets.mobilenet.conv_blocks import _fixed_padding, expand_input_by_factor, split_conv
import functools
import numpy as np
import argparse
import pickle
import re
import PIL
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

DEVICE = "CPU"
SPLIT_MODEL = False
EXPORT = True
SPLIT_EXPORT = True
IMPORT_PYTORCH = True

DUMP_QUANTIZE = True
QUANTIZE_LOCAL = True
IMAGENET_DIR = "" # Set to path to imagenet images if want to calibrate on imagenet
IMAGENET_IMGS = 20
LOCAL_DIR = ""# TODO: Set to path to jester calibration dataset (i.e. LOCAL_DIR/vid_num/img_*.jpg)
LOCAL_VIDS = 10

## Splits model for DPU such that shift op is offloaded to seperate partition for CPU
## expanded_conv_shift = CPU(Shift + Concat) -> DPU(Conv + ResidAdd)
##     concat_in, resid_in = CPU(prev_out)
##     out <- DPU(concat_in, resid_in)

top_arg_scope = tf.get_variable_scope()

torch_params_list = None
def torch_params(idx):
    global torch_params_list
    return torch_params_list[idx]

# Takes modulelist and returns parameter dict for linear layer
def torch_to_linear_state(module):
    init = tf.initializers.constant
    state = module.state_dict()

    linear_state = {
       "weights_initializer": init(state["weight"].numpy().T, verify_shape=True),
       "biases_initializer":  init(state["bias"].numpy(), verify_shape=True),
    }

    return linear_state


# Takes modulelist and returns parameter dict for Conv layer at index "i" and batchnorm at "i+1", and relu at "i+2"
def torch_to_conv_state(module, start_idx, depthwise=False):
    # Passing to slim.conv2d using tf.initializers.constant
    # weights_initializer = "0.weight"
    # bias_intiializer defaults to zero already
    # normalizer_params = {
    # scale: True,  ## enable gamma
    # center: True, ## enable beta
    # epsilon: mod.eps
    # param_initializers:
    #     {
    #     moving_mean: 1.running_mean,
    #     moving_var: 1.running_var,
    #     beta: 1.weights,
    #     gamma: 1.bias
    #     }
    # }
    conv_num = str(start_idx)
    bn_num = str(start_idx + 1)
    relu_num = str(start_idx + 2)

    init = tf.initializers.constant
    state = module.state_dict()

    def trans_w(x):
        if depthwise:
            # Pytorch weights are [out_C, in_C/groups, filter_H, filter_W].
            # TF weights are [filter_H, filter_W, in_C, channel_mult].
            # Assume in_C = groups, so channel_mult = in_C/groups
            return np.moveaxis(x, [0,1,2,3], [2,3,0,1])
        else:
            # Pytorch weights are [out_C, in_C, filter_H, filter_W].
            # TF weights are [filter_H, filter_W, in_C, out_C].
            return np.moveaxis(x, [0,1,2,3], [3,2,0,1])


    conv2d_init = {
        ## Conv2D
        "weights_initializer": init(trans_w(state[conv_num+".weight"].numpy()), verify_shape=True),
        "biases_initializer": tf.zeros_initializer(),

        ## Batch Norm
        "normalizer_params": {
            "scale": True,
            "center": True,
            "epsilon": module[int(bn_num)].eps,
            "param_initializers": {
                "moving_mean":       init(state[bn_num+".running_mean"].numpy(), verify_shape=True),
                "moving_variance":   init(state[bn_num+".running_var"].numpy(), verify_shape=True),
                "beta":              init(state[bn_num+".bias"].numpy(), verify_shape=True),
                "gamma":             init(state[bn_num+".weight"].numpy(), verify_shape=True),
            }
        }
    }

    return conv2d_init

# Stores initializers in torch_params_list
def import_pytorch_weights():
    import torch
    import mobilenet_v2_tsm_v2 as tsm_online
    global torch_params_list
    torch_params_list = []

    transpose = (DEVICE == "CPU") # Transpose pytorch weights (CHW) to TF-CPU (HWC)

    torch_model = tsm_online.MobileNetV2(n_class=27)
    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):  # checkpoint not downloaded
        print('Please run pytorch TSM online to download the model')
        return False

    torch_model.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar"))
    #torch_model.eval()

    i = 0
    for name,module in torch_model.named_modules():
        if isinstance(module, tsm_online.conv_bn) or isinstance(module, tsm_online.conv_1x1_bn):
            if isinstance(module, tsm_online.conv_bn):
                print("CONV_BN")
                print(module.state_dict()['1.running_mean'].numpy())
            else:
                print("CONV_1x1_BN")

            print("PNAMES: " + str(module.state_dict().keys()))
            params = torch_to_conv_state(module, 0)
            torch_params_list.append(params)
            i += 1
        elif isinstance(module, tsm_online.InvertedResidual) or isinstance(module, tsm_online.InvertedResidualWithShift):
            print("BOTTLENECK")
            layers = module.conv
            print(layers.state_dict().keys())
            print(len(layers))
            names = ["expansion_params", "depthwise_params", "projection_params"]

            # Exclude expansion if excluded in input
            name_idx = 1 if len(layers) < 8 else 0
            params = {}
            # Iterate conv, bn, relu (last relu optiona)
            for conv_idx in range(0, len(layers), 3):
                print(names[name_idx] + ": " + str(layers.state_dict()[str(conv_idx)+".weight"].numpy().shape))
                params[names[name_idx]] = torch_to_conv_state(layers, conv_idx, name_idx == 1)
                name_idx += 1
            torch_params_list.append(params)
            i += 1
        elif name == "classifier":
            print("CLASSIFIER")
            print(module.state_dict().keys())
            params = torch_to_linear_state(module)
            torch_params_list.append(params)
            i += 1

    return True

def export_quantize_info(out_path, inputs, output_node_names):
    with open(out_path, 'w') as f:
        in_shapes = []
        first = True
        f.write("--input_nodes \n")
        quote = False#len(inputs) > 1
        if quote:
            f.write("\"")
        for i,(tensor,array) in enumerate(inputs.items()):
            name = tensor
            if type(tensor) != str:
                name = tensor.name

            # Skip CPU shift buffer
            if SPLIT_MODEL and "shift_buffer" in name:
                continue
            node = name.split(":0")[0]
            in_shapes.append(array.shape)

            if not first:
                f.write(",")
            first = False
            f.write(node)
        if quote:
            f.write("\"")
        f.write("\n\n")

        first = True
        f.write("--input_shapes \n")
        quote = False#len(in_shapes) > 1
        if quote:
            f.write("\"")
        for i,shape in enumerate(in_shapes):
            if not first:
                f.write(":")
            first = False
            f.write(f"{shape[0]},{shape[1]},{shape[2]},{shape[3]}")
        if quote:
            f.write("\"")
        f.write("\n\n")

        first = True
        f.write("--output_nodes \n")
        quote = False#len(output_node_names) > 1
        if quote:
            f.write("\"")
        for i,out in enumerate(output_node_names):
            # Skip CPU split and concat outputs
            if "shift_split_buffer_output" in out or "shift_concat_output" in out:
                continue

            if not first:
                f.write(",")
            first = False
            f.write(out)
        if quote:
            f.write("\"")

# Based on ops.expanded_conv
# Changes wrapped in ### SHIFT CHANGE ###
@slim.add_arg_scope
def expanded_conv_shift(input_tensor,
                  num_outputs,
                  shift_buffer_name="", ### SHIFT_CHANGE ###
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  split_projection=1,
                  split_expansion=1,
                  split_divisible_by=8,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  inner_activation_fn=None,
                  depthwise_activation_fn=None,
                  project_activation_fn=tf.identity,
                  depthwise_fn=slim.separable_conv2d,
                  expansion_fn=split_conv,
                  projection_fn=split_conv,
                  depthwise_params=None,
                  expansion_params=None,
                  projection_params=None,
                  scope=None):
  """Depthwise Convolution Block with expansion.

  Builds a composite convolution that has the following structure
  expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

  Args:
    input_tensor: input
    num_outputs: number of outputs in the final layer.
    expansion_size: the size of expansion, could be a constant or a callable.
      If latter it will be provided 'num_inputs' as an input. For forward
      compatibility it should accept arbitrary keyword arguments.
      Default will expand the input by factor of 6.
    stride: depthwise stride
    rate: depthwise rate
    kernel_size: depthwise kernel
    residual: whether to include residual connection between input
      and output.
    normalizer_fn: batchnorm or otherwise
    split_projection: how many ways to split projection operator
      (that is conv expansion->bottleneck)
    split_expansion: how many ways to split expansion op
      (that is conv bottleneck->expansion) ops will keep depth divisible
      by this value.
    split_divisible_by: make sure every split group is divisible by this number.
    expansion_transform: Optional function that takes expansion
      as a single input and returns output.
    depthwise_location: where to put depthwise covnvolutions supported
      values None, 'input', 'output', 'expansion'
    depthwise_channel_multiplier: depthwise channel multiplier:
    each input will replicated (with different filters)
    that many times. So if input had c channels,
    output will have c x depthwise_channel_multpilier.
    endpoints: An optional dictionary into which intermediate endpoints are
      placed. The keys "expansion_output", "depthwise_output",
      "projection_output" and "expansion_transform" are always populated, even
      if the corresponding functions are not invoked.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    padding: Padding type to use if `use_explicit_padding` is not set.
    inner_activation_fn: activation function to use in all inner convolutions.
    If none, will rely on slim default scopes.
    depthwise_activation_fn: activation function to use for deptwhise only.
      If not provided will rely on slim default scopes. If both
      inner_activation_fn and depthwise_activation_fn are provided,
      depthwise_activation_fn takes precedence over inner_activation_fn.
    project_activation_fn: activation function for the project layer.
    (note this layer is not affected by inner_activation_fn)
    depthwise_fn: Depthwise convolution function.
    expansion_fn: Expansion convolution function. If use custom function then
      "split_expansion" and "split_divisible_by" will be ignored.
    projection_fn: Projection convolution function. If use custom function then
      "split_projection" and "split_divisible_by" will be ignored.
    depthwise_params: kwargs to pass to depthwise_fn.
    expansion_params: kwargs to pass to expansion_fn.
    projection_params: kwargs to pass to projection_fn.

    scope: optional scope.

  Returns:
    Tensor of depth num_outputs

  Raises:
    TypeError: on inval
  """
  conv_defaults = {}
  dw_defaults = {}
  if inner_activation_fn is not None:
    conv_defaults['activation_fn'] = inner_activation_fn
    dw_defaults['activation_fn'] = inner_activation_fn
  if depthwise_activation_fn is not None:
    dw_defaults['activation_fn'] = depthwise_activation_fn
  # pylint: disable=g-backslash-continuation
  with tf.variable_scope(scope, default_name='expanded_conv_shift') as s, \
       tf.name_scope(s.original_name_scope), \
      slim.arg_scope((slim.conv2d,), **conv_defaults), \
       slim.arg_scope((slim.separable_conv2d,), **dw_defaults):
    prev_depth = input_tensor.get_shape().as_list()[3]
    if  depthwise_location not in [None, 'input', 'output', 'expansion']:
      raise TypeError('%r is unknown value for depthwise_location' %
                      depthwise_location)
    if use_explicit_padding:
      if padding != 'SAME':
        raise TypeError('`use_explicit_padding` should only be used with '
                        '"SAME" padding.')
      padding = 'VALID'
    depthwise_func = functools.partial(
        depthwise_fn,
        num_outputs=None,
        kernel_size=kernel_size,
        depth_multiplier=depthwise_channel_multiplier,
        stride=stride,
        rate=rate,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope='depthwise',
        **depthwise_params)
    # b1 -> b2 * r -> b2
    #   i -> (o * r) (bottleneck) -> o
    #input_tensor = tf.identity(input_tensor, 'input')
    #net = input_tensor

    ### SHIFT CHANGES ###
    # If splitting the model for DPU, insert placeholder input for feeding
    if SPLIT_MODEL:
        net = tf.identity(input_tensor, 'prev_conv_output')
        input_tensor = tf.placeholder(tf.float32, shape=input_tensor.get_shape(), name='input')
    else:
        input_tensor = tf.identity(input_tensor, 'input')
    ### END SHIFT CHANGES ###
    net = input_tensor

    ### SHIFT CHANGES ###
    # implements the shift operation
    c = input_tensor.shape[3].value
    x1, x2 = tf.split(input_tensor, [c // 8, c - (c // 8)], axis=3, num=2, name='shift_split')
    shift_out = tf.identity(x1, 'shift_split_buffer_output')
    #x2 = tf.slice(input_tensor, tf.constant([0,0,0,c//8]),
    #              tf.constant([input_tensor.shape[0].value, input_tensor.shape[1].value, input_tensor.shape[2].value, c-c//8]))#input_tensor[:, :, :, c//8:]
    shift_buffer = tf.get_default_graph().get_tensor_by_name(shift_buffer_name)
    shift_concat = tf.concat((shift_buffer, x2), axis=3, name='shift_concat')
    ### END SHIFT CHANGES ###

    ### SHIFT CHANGES ###
    # If splitting, (concat, input_tensor) are fed to DPU. Insert new placeholder for concat input
    if SPLIT_MODEL:
        _ = tf.identity(shift_concat, 'shift_concat_output')
        shift_concat = tf.placeholder(tf.float32, shape=shift_concat.get_shape(), name='shift_concat_input')
    ### END SHIFT CHANGES ###
    net = shift_concat

    if depthwise_location == 'input':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net, activation_fn=None)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if callable(expansion_size):
      inner_size = expansion_size(num_inputs=prev_depth)
    else:
      inner_size = expansion_size

    if inner_size > net.shape[3]:
      if expansion_fn == split_conv:
        expansion_fn = functools.partial(
            expansion_fn,
            num_ways=split_expansion,
            divisible_by=split_divisible_by,
            stride=1)
      net = expansion_fn(
          net,
          inner_size,
          scope='expand',
          normalizer_fn=normalizer_fn,
          **expansion_params)
      net = tf.identity(net, 'expansion_output')
      if endpoints is not None:
        endpoints['expansion_output'] = net

    if depthwise_location == 'expansion':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if expansion_transform:
      net = expansion_transform(expansion_tensor=net, input_tensor=input_tensor)
    # Note in contrast with expansion, we always have
    # projection to produce the desired output size.
    if projection_fn == split_conv:
      projection_fn = functools.partial(
          projection_fn,
          num_ways=split_projection,
          divisible_by=split_divisible_by,
          stride=1)
    net = projection_fn(
        net,
        num_outputs,
        scope='project',
        normalizer_fn=normalizer_fn,
        activation_fn=project_activation_fn,
        **projection_params)
    if endpoints is not None:
      endpoints['projection_output'] = net
    if depthwise_location == 'output':
      if use_explicit_padding:
        net = _fixed_padding(net, kernel_size, rate)
      net = depthwise_func(net, activation_fn=None)
      net = tf.identity(net, name='depthwise_output')
      if endpoints is not None:
        endpoints['depthwise_output'] = net

    if callable(residual):  # custom residual
      net = residual(input_tensor=input_tensor, output_tensor=net)
    elif (residual and
          # stride check enforces that we don't add residuals when spatial
          # dimensions are None
          stride == 1 and
          # Depth matches
          net.get_shape().as_list()[3] ==
          input_tensor.get_shape().as_list()[3]):
      net += input_tensor
    return tf.identity(net, name='output')



op = lib.op
expand_input = ops.expand_input_by_factor


### Import pytorch config before setting constant initializers through torch_params
if IMPORT_PYTORCH and not import_pytorch_weights():
    sys.stderr.write("Error importing pytorch weights\n")
    sys.exit(1)

# Based on V2_DEF from slim mobilenet_v2
V2_DEF_TSM = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6,
        },
        (ops.expanded_conv,expanded_conv_shift): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'},
        #(slim.conv2d, slim.separable_conv2d): {'biases_initializer': tf.initializers.constant(0.01)} ### SHIFT CHANGE to allow untrained inference for testing ###
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3], **torch_params(0)),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16,
           **torch_params(1)
          ),
        op(ops.expanded_conv,   stride=2, num_outputs=24, **torch_params(2)),
        op(expanded_conv_shift, stride=1, num_outputs=24, shift_buffer_name='shift_buffer_0:0', **torch_params(3)), # Shift
        op(ops.expanded_conv,   stride=2, num_outputs=32, **torch_params(4)),
        op(expanded_conv_shift, stride=1, num_outputs=32, shift_buffer_name='shift_buffer_1:0', **torch_params(5)), # Shift
        op(expanded_conv_shift, stride=1, num_outputs=32, shift_buffer_name='shift_buffer_2:0', **torch_params(6)), # Shift
        op(ops.expanded_conv,   stride=2, num_outputs=64, **torch_params(7)),
        op(expanded_conv_shift, stride=1, num_outputs=64, shift_buffer_name='shift_buffer_3:0', **torch_params(8)), # Shift
        op(expanded_conv_shift, stride=1, num_outputs=64, shift_buffer_name='shift_buffer_4:0', **torch_params(9)), # Shift
        op(expanded_conv_shift, stride=1, num_outputs=64, shift_buffer_name='shift_buffer_5:0', **torch_params(10)), # Shift
        op(ops.expanded_conv,   stride=1, num_outputs=96, **torch_params(11)),
        op(expanded_conv_shift, stride=1, num_outputs=96, shift_buffer_name='shift_buffer_6:0', **torch_params(12)), # Shift
        op(expanded_conv_shift, stride=1, num_outputs=96, shift_buffer_name='shift_buffer_7:0', **torch_params(13)), # Shift
        op(ops.expanded_conv,   stride=2, num_outputs=160, **torch_params(14)),
        op(expanded_conv_shift, stride=1, num_outputs=160, shift_buffer_name='shift_buffer_8:0', **torch_params(15)), # Shift
        op(expanded_conv_shift, stride=1, num_outputs=160, shift_buffer_name='shift_buffer_9:0', **torch_params(16)), # Shift
        op(ops.expanded_conv,   stride=1, num_outputs=320, **torch_params(17)),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280, **torch_params(18))
    ],
)

checkpoint = "mobilenet_v2_1.0_224.ckpt"
tf.reset_default_graph()


shift_buffer_shapes = [[1, 56, 56, 3],
                      [1, 28, 28, 4],
                      [1, 28, 28, 4],
                      [1, 14, 14, 8],
                      [1, 14, 14, 8],
                      [1, 14, 14, 8],
                      [1, 14, 14, 12],
                      [1, 14, 14, 12],
                      [1, 7, 7, 20],
                      [1, 7, 7, 20]]

#with tf.variable_scope("Mobilenet", "Mobilenet") as scope, tf.Session() as sess:
#shift_buffer = [tf.placeholder(tf.float32, shape=[1, 3, 56, 56], name='shift_buffer_0'),
#                tf.placeholder(tf.float32, shape=[1, 4, 28, 28], name='shift_buffer_1'),
#                tf.placeholder(tf.float32, shape=[1, 4, 28, 28], name='shift_buffer_2'),
#                tf.placeholder(tf.float32, shape=[1, 8, 14, 14], name='shift_buffer_3'),
#                tf.placeholder(tf.float32, shape=[1, 8, 14, 14], name='shift_buffer_4'),
#                tf.placeholder(tf.float32, shape=[1, 8, 14, 14], name='shift_buffer_5'),
#                tf.placeholder(tf.float32, shape=[1, 12, 14, 14], name='shift_buffer_6'),
#                tf.placeholder(tf.float32, shape=[1, 12, 14, 14], name='shift_buffer_7'),
#                tf.placeholder(tf.float32, shape=[1, 20, 7, 7], name='shift_buffer_8'),
#                tf.placeholder(tf.float32, shape=[1, 20, 7, 7], name='shift_buffer_9')]
shift_buffer = [tf.placeholder(tf.float32, shape=shift_buffer_shapes[0], name='shift_buffer_0'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[1], name='shift_buffer_1'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[2], name='shift_buffer_2'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[3], name='shift_buffer_3'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[4], name='shift_buffer_4'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[5], name='shift_buffer_5'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[6], name='shift_buffer_6'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[7], name='shift_buffer_7'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[8], name='shift_buffer_8'),
                tf.placeholder(tf.float32, shape=shift_buffer_shapes[9], name='shift_buffer_9')]

#FINAL_NODE_NAME="MobilenetV2/Conv_1/Relu6"
FINAL_NODE_NAME="MobilenetV2/Logits/output"

in_tensor = tf.placeholder(tf.float32, shape=(1,224,224,3), name='in_img')

print(torch_params(0)['normalizer_params']['param_initializers']['moving_mean'].get_config())

in_img = tf.identity(in_tensor)

net, endpoints = mobilenet_v2.mobilenet_base(in_img, conv_defs=V2_DEF_TSM)

# Add the classifier
with tf.variable_scope("MobilenetV2/Logits"):
    kernel_initializer = None
    bias_initializer = tf.zeros_initializer()
    if IMPORT_PYTORCH:
        kernel_initializer = torch_params(-1)["weights_initializer"]
        bias_initializer  = torch_params(-1)["biases_initializer"]

    net = tf.nn.avg_pool(net, [1,7,7,1], 1, "VALID", name="AvgPool")
    net = tf.squeeze(net, (1,2))
    net = tf.layers.dense(net, 27, use_bias=True, trainable = False,
                       kernel_initializer = kernel_initializer,
                       bias_initializer  = bias_initializer,
                       name="Linear")
    #net = tf.layers.Conv2D(27, [1,1], 
    #                  kernel_initializer = kernel_initializer,
    #                  bias_initializer  = bias_initializer,
    #                  name="Linear")(net)
    #net = tf.keras.layers.Dense(27, use_bias=True,
    #                   kernel_initializer = kernel_initializer,
    #                   bias_initializer  = bias_initializer,
    #                   name="Linear")(net)

    net = tf.identity(net, name="output")

#ema = tf.train.ExponentialMovingAverage(0.999)
#vars = ema.variables_to_restore()
#saver = tf.train.Saver(vars)

split_outputs = []
output_node_names = []
inputs = {}
frozen_graph_def = None

with tf.Session() as sess:
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    print(tf.global_variables()[3])
    print(tf.global_variables()[3].initializer)
    print("GLOBALS: " + str(tf.global_variables()))

    print("NODES: " + str([x.name for x in graph.as_graph_def().node]))

    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "../tf_models/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")

    print([n.name for n in graph.get_operations()])
    output_node_names = [FINAL_NODE_NAME]

    # add shift buffer output nodes
    for op in graph.get_operations():
        if "Identity" in op.type and "shift_split_buffer_output" in op.name:
            output_node_names.append(op.name)

    inputs = {in_tensor: np.ones((1,224,224,3))}
    for i,shape in enumerate(shift_buffer_shapes):
        inputs[shift_buffer[i]] = np.zeros(shape)

    # update outputs to include buffer shift outputs and inputs to expanded_conv_shift layers
    # update inputs to include internal placeholders
    if SPLIT_MODEL:
        for op in graph.get_operations():
            # CPU input for shift layer from previous layer
            if "Identity" in op.type and "prev_conv_output" in op.name:
                output_node_names.append(op.name)
            # CPU -> DPU concat output
            if "Identity" in op.type and "shift_concat_output" in op.name:
                output_node_names.append(op.name)
            # CPU input for shift
            if op.type == "Placeholder" and "/input" in op.name:
                inputs[op.name+':0'] = np.ones(op.outputs[0].get_shape())
            # DPU input for conv
            if op.type == "Placeholder" and "shift_concat_input" in op.name:
                inputs[op.name+':0'] = np.ones(op.outputs[0].get_shape())

    ## Dump split inputs to pickle file for quantization
    if DUMP_QUANTIZE:
        assert not SPLIT_MODEL

        input_dir = LOCAL_DIR if QUANTIZE_LOCAL else IMAGENET_DIR
        assert os.path.isdir(input_dir)

        inters = ["in_img:0"]
        input_dict = {0: len(graph.get_operations())}
        shift_in_dict = {0: 0} # No shift buffer in first layer. Dummy input
        shift_out_dict = {}
        for op in graph.get_operations():
            in_search = re.search("shift(_(\d+))?/input$", op.name)
            shift_in_search = re.search("shift(_(\d+))?/shift_concat$", op.name)
            shift_out_search = re.search("shift(_(\d+))?/shift_split_buffer_output$", op.name)
            if "Identity" in op.type and in_search:
                n = 0
                if in_search.group(1):
                    n = int(in_search.group(1)[1:])
                input_dict[n+1] = len(inters)
                inters.append(op.name + ":0")
            elif shift_in_search:
                n = 0
                if shift_in_search.group(1):
                    n = int(shift_in_search.group(1)[1:])
                shift_in_dict[n+1] = len(inters)
                inters.append(op.name + ":0")
            elif shift_out_search:
                n = 0
                if shift_out_search.group(1):
                    n = int(shift_out_search.group(1)[1:])
                shift_out_dict[n] = len(inters)
                inters.append(op.name + ":0")

        print("DICTS:")
        print(input_dict.keys())
        print(shift_in_dict.keys())
        print(shift_out_dict.keys())
        #print("INTERS: " + str(inters))


        inputs = {}

        shift_data = []
        for shape in shift_buffer_shapes:
            shift_data.append(np.zeros(shape))

        img_paths = []
        if QUANTIZE_LOCAL:
            for vid in sorted(os.listdir(input_dir))[:LOCAL_VIDS]:
                vid_path = os.path.join(input_dir, vid)
                num_frames = len(os.listdir(vid_path))
                print(f"Vid {vid} has {num_frames} frames", file=sys.stderr)
                for img in sorted(os.listdir(vid_path)):
                    img_paths.append(os.path.join(vid_path, img))
        else:
            img_paths = [os.path.join(input_dir, x) for x in sorted(os.listdir(input_dir))[:IMAGENET_IMGS]]
        dump_data = {}
        for img_num,p_img in enumerate(img_paths):
            print(f"Processing calib data # {img_num}...", file=sys.stderr)
            img = PIL.Image.open(p_img)

            w,h = img.size
            new_w = 0
            new_h = 0
            if w > h:
                new_h = 256
                new_w = (256*w)//h
            else:
                new_w = 256
                new_h = (256*h)//w
            img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
            left = (new_w - 224)//2
            top = (new_h - 224)//2
            img = img.crop((left, top, left+224, top+224))
            img = np.array(img)/255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            inputs[in_tensor] = np.expand_dims(img, axis=0)
            for i in range(len(shift_buffer_shapes)):
                inputs[shift_buffer[i]] = shift_data[i]

            outputs = sess.run(inters, inputs)

            if QUANTIZE_LOCAL:
                for i in range(len(shift_out_dict)):
                    shift_data[i] = outputs[shift_out_dict[i]]

            assert len(input_dict) == len(shift_in_dict)
            for i in range(len(input_dict)):
                shift = outputs[shift_in_dict[i]] if shift_in_dict[i] < len(outputs) else np.empty(1)
                resid = outputs[input_dict[i]] if input_dict[i] < len(outputs) else np.empty(1)
                if img_num == 0:
                    dump_data[i] = {}
                dump_data[i][img_num] = {"shift_concat": shift.tolist(),
                                         "resid": resid.tolist()}

        print(f"Dumping Quantize Data...", file=sys.stderr)
        for split_num in range(len(input_dict)):
            with open(os.path.join("model_tf_split_export", f"model_tf_split_{split_num}", "inputs.pickle"), 'wb') as f:
                pickle.dump(dump_data[split_num], f, pickle.HIGHEST_PROTOCOL)

    if EXPORT:
        print(f"Saving model...", file=sys.stderr)

        print("INS: " + str(inputs.keys()))
        print("OUTS: " + str(output_node_names))
        sess.run([o + ":0" for o in output_node_names], inputs)


        for op in graph.get_operations():
            if "Identity" in op.type and "prev_conv_output" in op.name:
                split_outputs.append(op.name)
        split_outputs.append(FINAL_NODE_NAME)

        saver = tf.train.Saver()

        model_name = "model_tf_split" if SPLIT_MODEL else "model_tf"
        save_dir = os.path.join(".", model_name)

        print(f"Saving model to {save_dir}...")
        ckpt_file = saver.save(sess, os.path.join(save_dir, model_name + ".ckpt"))
        pbtxt_file = model_name + ".pbtxt"
        tf.train.write_graph(graph_or_graph_def=input_graph_def, logdir=save_dir, name=pbtxt_file, as_text=True)

        print("IN_TEST: " + str(inputs.keys()))
        export_quantize_info(os.path.join(save_dir, "quantize_info.txt"), inputs, output_node_names)

        pbtxt_path = os.path.join(save_dir, pbtxt_file)
        pb_path = os.path.join(save_dir, model_name + ".pb")
        frozen_graph_def = freeze_graph.freeze_graph(input_graph=pbtxt_path, input_saver='', input_binary=False, input_checkpoint=ckpt_file, output_node_names=",".join(output_node_names), output_graph=pb_path, restore_op_name="save/restore_all", filename_tensor_name="save/Const:0", clear_devices=True, initializer_nodes="")


print("DONE")

### Save a frozen graph for each disconnected portion
if EXPORT and SPLIT_MODEL and SPLIT_EXPORT:
    base_dir = "model_tf_split_export"
    print(f"Exporting split graphs to {base_dir}...")
    for split_num,out in enumerate(split_outputs):
        tf.reset_default_graph()

        output_node_names = []
        inputs = {}

        split_graph_def = tf.graph_util.extract_sub_graph(frozen_graph_def, [out])
        with tf.Session() as split_sess:
            tf.graph_util.import_graph_def(split_graph_def, name="")
            split_graph = tf.get_default_graph()

            # We are the input split
            if split_num == 0:
                inputs = {in_tensor: np.ones((1,224,224,3))}

            for op in split_graph.get_operations():
                ## INPUTS
                # shift buffer input
                if split_num > 0:
                    inputs[shift_buffer[split_num-1]] = np.ones(shift_buffer_shapes[split_num-1])
                # CPU input for shift
                if op.type == "Placeholder" and "/input" in op.name:
                    inputs[op.name+':0'] = np.ones(op.outputs[0].get_shape())
                # DPU input for conv
                elif op.type == "Placeholder" and "shift_concat_input" in op.name:
                    inputs[op.name+':0'] = np.ones(op.outputs[0].get_shape())

                ## OUTPUTS
                # Model output
                if FINAL_NODE_NAME in op.name:
                    output_node_names.append(op.name)
                # Shift buffer output
                elif "Identity" in op.type and "shift_split_buffer_output" in op.name:
                    output_node_names.append(op.name)
                # CPU input for shift layer from previous layer
                elif "Identity" in op.type and "prev_conv_output" in op.name:
                    output_node_names.append(op.name)
                # CPU -> DPU concat output
                elif "Identity" in op.type and "shift_concat_output" in op.name:
                    output_node_names.append(op.name)



        model_name = f"model_tf_split_{split_num}"
        save_dir = os.path.join(".", base_dir, model_name)
        pb_path = os.path.join(save_dir, model_name + ".pb")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with gfile.GFile(pb_path, "wb") as f:
            f.write(split_graph_def.SerializeToString())
        export_quantize_info(os.path.join(save_dir, "quantize_info.txt"), inputs, output_node_names)
