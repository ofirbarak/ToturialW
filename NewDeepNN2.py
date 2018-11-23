from AEconstants import *

K = [4, 4]  # sparse size
M_O = 4  # output batch size, assume LAYERS modulo M_O = 0
M_I = 4  # input batch size

import tensorflow as tf
import numpy as np


def regular_conv(input, kernel, biases, stride, batch_size, trans=False, out_shape=None):
    """
    preform regular convolution
    :param input: a 4D tensor object
    :param kernel: a 4D tensor - if trans=False [H,W,I_C,O_C], else [H,W,O_C,I_C]
    :param biases: bias, 1D tensor
    :param s_h: stride's size - [stride, stride]
    :param padding: padding algorithm
    :param trans: if true preform conv_transpose, otherwise regular convolution
    :param out_shape: in transpose conv, a 2D array - [height, width] of the output image
    :return: convolution result
    """
    # batch_size = tf.shape(input)[0]
    padding = "SAME"
    # if trans:
    #     size = kernel.shape.as_list()
    #     conv = tf.nn.conv2d_transpose(input, kernel, [batch_size]+out_shape+[size[-2]], [1, stride, stride, 1], padding)
    #     return tf.nn.bias_add(conv, biases)
    # else:
    conv = None
    if trans:
        # kernel = tf.transpose(kernel, [0, 1, 3, 2])
        size = kernel.shape.as_list()
        conv = tf.nn.conv2d_transpose(input, kernel ,[batch_size]+out_shape+[size[-2]], [1,stride,stride,1], padding)
    else:
        conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding)
    return tf.nn.bias_add(conv, biases)


def prepare_buf(buffers, k):
    buf_val, buf_ind, buf_loc = buffers
    old_buf_len = len(buf_val)
    if k == old_buf_len:
        return buf_val, buf_ind, buf_loc

    nbuf_val = [tf.zeros_like(buf_val[0]) for _ in range(k)]
    nbuf_ind = [tf.zeros_like(buf_ind[0]) for _ in range(k)]
    nbuf_loc = [tf.zeros_like(buf_loc[0]) for _ in range(k)]

    if old_buf_len > k:
        # shrink buf
        # copy from buf_val and buf_ind the k highest values
        for i in range(k):
            nbuf_val[i] = tf.reduce_max(buf_val)
            ind_v = tf.argmax(buf_val)

            # update buf_val, and nbuf_ind
            for l, buf_layer in enumerate(buf_val):
                mask = tf.equal(ind_v, l)
                maskf = tf.cast(mask, tf.float32)
                maski = tf.cast(mask, tf.int32)
                buf_val[l] = (1-maskf)*buf_val[l] # put zeros where we updated
                nbuf_ind[i] = (1-maski)*nbuf_ind[i] + maski*buf_ind[l]
                nbuf_loc[i] = (1-maski)*nbuf_loc[i] + maski*buf_loc[l]

    else:
        # expand buf
        for i in range(len(buf_val)):
            nbuf_val[i] = buf_val[i]
            nbuf_ind[i] = buf_ind[i]
            nbuf_loc[i] = buf_loc[i]

    return (nbuf_val, nbuf_ind, nbuf_loc)



def update_buf(buffers, responses, location, filter_init_num, k):
    """
    Update buffer according to the new responses
    :param buf_val:  list of length k, of tensors of shape as conv result, 2d
    :param buf_ind:  list of length k, of tensors of shape as conv result, 2d
    :param responses: a tensor of shape (1, resulted_img_size(2 dims), M_O)
    :param filter_init_num: integer that indicates the start filter's number
    :return: updated buffer
    """
    # print(location)
    responses_shape = responses.shape.as_list()

    # create an filter's index array for buf_ind updates
    filter_num = np.arange(filter_init_num, filter_init_num+responses_shape[-1])

    # check old buffer size against k and update buffers
    buf_val, buf_ind, buf_loc = prepare_buf(buffers, k)

    for j in range(responses.shape[-1]):
        r_layer = responses[:,:,j]

        tensor_buf_val = tf.stack(buf_val, -1)
        mmin = tf.reduce_min(tensor_buf_val, -1)
        mind = tf.argmin(tensor_buf_val, -1)
        mask = tf.greater(r_layer, mmin)

        for i, buf_layer in enumerate(buf_val):
            tm = tf.logical_and(mask, tf.equal(mind, i))
            tmf = tf.cast(tm, tf.float32)
            tmi = tf.cast(tm, tf.int32)
            buf_val[i] = (1.0-tmf)*buf_val[i] + tmf*r_layer
            buf_ind[i] = (1-tmi)*buf_ind[i] + tmi*filter_num[j]
            buf_loc[i] = (1-tmi)*buf_loc[i] + tmi*location[:,:,j]


        # # iterate the buffer, and update in the rights placed in each slice
        # for i, buf_layer in enumerate(buf_val):
        #     mask = tf.greater(r_layer, buf_layer)
        #     mask = tf.cast(mask, tf.float32)
        #     buf_val[i] = buf_layer*(1.0-mask) + mask*r_layer
        #     buf_ind[i] = buf_ind[i]*(1-mask) + mask*filter_num[j]
        #     r_layer = r_layer*(1-mask) + 0*mask


        # ---- OLD VERSION --
        #
        # # update buffers
        # i1,i2 = np.indices(buf_val.shape.as_list()[:-1])
        # indexes = tf.stack([i1.flatten(), i2.flatten(), tf.reshape(mind, [-1])] ,axis=-1)
        # indices = tf.boolean_mask(indexes, tf.reshape(mask, [-1]))
        #
        # val_to_update2d = tf.boolean_mask(max_v, mask)
        # ind_to_update2d = tf.boolean_mask(ind, mask)
        #
        # a,b,c = np.indices(buf_val.shape.as_list())
        # indexes3d = tf.stack([a.flatten(),b.flatten(),c.flatten()], axis=-1)
        #
        # ind_mask = tf.equal(indexes3d, indices)
        # ind_mask = tf.reduce_all(ind_mask, axis=-1)
        # # print(indexes3d.shape, ind_mask.shape, buf_val.shape)
        # ind_mask = tf.reshape(ind_mask, buf_val.shape)
        # ind_mask = tf.cast(ind_mask, buf_val.dtype)
        #
        # val_to_update3d = tf.SparseTensor(indices, val_to_update2d, buf_val.shape)
        # ind_to_update3d = tf.SparseTensor(indices, ind_to_update2d, buf_val.shape)
        #
        # val_to_update3d = tf.cast(val_to_update3d, buf_val.dtype)
        # ind_to_update3d = tf.cast(ind_to_update3d, buf_val.dtype)
        #
        # buf_val = tf.sparse_add(val_to_update3d, buf_val*(1-ind_mask))
        # buf_ind = tf.sparse_add(ind_to_update3d, buf_val*(1-ind_mask))

    return buf_val, buf_ind, buf_loc


def update_buffers(buffers, responses, locations, filter_init_num, k):
    # print(locations)
    resp_shape = responses.get_shape().as_list()
    # offset = resp_shape[0]*resp_shape[1]*resp_shape[2]*resp_shape[3]*filter_init_num
    offset = 0
    # print('update buffers', resp_shape, filter_init_num, offset)
    locations = locations + offset
    # print('update', offset)
    ubuffers = []
    batch_size = len(buffers)
    for i in range(batch_size):
        ubuffers.append(update_buf(buffers[i], responses[i], locations[i],filter_init_num, k))

    return ubuffers


def reconstruct(buffers, i, m):
    """
    Reconstruct an input (image)
    :param buffers: buf_ind and buf_val buffers
    :param i: initial channel to restore
    :param m: size of channels to restore
    :return: the reconstruct input
    """
    buf_val, buf_ind, buf_loc = buffers
    I = []
    relevant_loc = []
    tensor_buf_ind = tf.stack(buf_ind, -1)
    tensor_buf_val = tf.stack(buf_val, -1)

    tensor_buf_loc = None
    if type(buf_loc) == list:
        tensor_buf_loc = tf.stack(buf_loc, -1)
    else:
        tensor_buf_loc = buf_loc

    for j in range(i, i + m):
        mask = tf.equal(tensor_buf_ind, tf.constant(j, dtype=tensor_buf_ind.dtype))
        mask = tf.cast(mask, tensor_buf_val.dtype)
        maskr = tf.cast(mask, tensor_buf_loc.dtype)
        I.append(tf.reduce_sum(tensor_buf_val * mask, -1))
        relevant_loc.append(tf.reduce_sum(tensor_buf_loc * maskr, -1)) #todo: problem with mask

    return tf.expand_dims(tf.stack(I, -1), axis=0), tf.stack(relevant_loc, -1)


def reconstruct_batch(buffers, i, m):
    """
    Reconstruct batch od images
    :param buffers: list of tuples (buf_val, buf_ind) for each image
    :param i: initial channel to restore
    :param m: size of channels to restore
    :return: the reconstruct batch
    """
    batch = []
    batch_loc = []
    batch_size = len(buffers)
    for ind in range(batch_size):
        I, loc = reconstruct(buffers[ind], i, m)
        batch.append(I[0])
        batch_loc.append(loc)

    batch = tf.convert_to_tensor(batch)
    buf_loc = tf.convert_to_tensor(batch_loc)

    # img_shape = batch.get_shape().as_list()
    # offset = img_shape[0]*img_shape[1]*img_shape[2]*img_shape[3]*i
    # print('reconstruct offest', offset)
    # offset = 0
    # print('reconstruct batch', img_shape, i, offset)
    return batch, buf_loc
    # return batch, buf_loc #tf.maximum(buf_loc-offset, 0)


def unpool(I, locations, size):
    if type(locations) == list:
        locations = tf.convert_to_tensor(locations)
    print('unpool', locations)
    input_shape = I.get_shape().as_list()
    batch = []
    for ind in range(input_shape[0]):
        flatten_argmax = tf.reshape(locations[ind], [-1,1])
        flatten_maxval = tf.reshape(I[ind], [-1])

        # flatten_argmax = []
        # flatten_maxval = []
        # for ch in range(input_shape[-1]):
        #     flatten_argmax.append(tf.reshape(locations[ind, :, :, ch], [-1]))
        #     flatten_maxval.append(tf.reshape(I[ind, :, :, ch], [-1]))
        # tensor_argmax = tf.stack(flatten_argmax, 0)
        # tensor_maxval = tf.stack(flatten_maxval, 0)
        # print(tensor_argmax, tensor_maxval)
        # flatten_argmax = tf.reshape(tensor_argmax, [-1, 1])
        # flatten_maxval = tf.reshape(tensor_maxval, [-1])
        # print(flatten_argmax, flatten_maxval)

        # print(flatten_argmax, flatten_maxval, I)
        sh = tf.constant([1 * int(size[0]) * int(size[1]) * input_shape[-1]], dtype=flatten_argmax.dtype)
        batch_maxpoollike = tf.scatter_nd(flatten_argmax, flatten_maxval, sh)
        img = tf.reshape(batch_maxpoollike, (size[0], size[1], input_shape[-1]))
        batch.append(img)

    return tf.stack(batch, 0)

    # input_shape = I.get_shape().as_list()
    # flatten_argmax = tf.reshape(locations, [-1, 1])
    # flatten_maxval = tf.reshape(I, [-1])
    # sh = tf.constant([input_shape[0] * int(size[0]) * int(size[1]) * input_shape[-1]], dtype=flatten_argmax.dtype)
    # batch_maxpoollike = tf.scatter_nd(flatten_argmax, flatten_maxval, sh)
    # img = tf.reshape(batch_maxpoollike, (input_shape[0], size[0], size[1], input_shape[-1]))
    # return img

    # ###### Other Impelementation ##################
    # print(I, locations)
    # net = I
    # mask = locations
    # # print('ind', ind)
    # input_shape = net.get_shape().as_list()
    # output_shape = (batch_size, int(size[0]), int(size[1]), input_shape[-1])
    #
    #
    # # calculation indices for batch, height, width and feature maps
    # one_like_mask = tf.ones_like(mask)
    # batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=[input_shape[0], 1, 1, 1])
    # b = one_like_mask * batch_range
    # y = mask // (output_shape[2] * output_shape[3])
    # x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    # feature_range = tf.range(output_shape[3], dtype=tf.int32)
    # f = one_like_mask * feature_range
    # # transpose indices & reshape update values to one dimension
    # updates_size = tf.size(net)
    # indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    # values = tf.reshape(net, [updates_size])
    # ret = tf.scatter_nd(indices, values, output_shape)
    # return ret

def get_maxpool_argmax(batch_responses, window, stride):
    responses_loc = []
    for i in range(batch_responses.shape[0]):
        _, resp_loc = tf.nn.max_pool_with_argmax(batch_responses[i:i+1,:,:,:], [1, window, window, 1],
                                                 [1, stride, stride, 1], "SAME")
        resp_loc = tf.stop_gradient(resp_loc)
        resp_loc = tf.cast(resp_loc, tf.int32)
        responses_loc.append(resp_loc[0])

    batch_responses = tf.nn.max_pool(batch_responses, [1, window, window, 1],
                                     [1, stride, stride, 1], padding="SAME")

    return batch_responses, tf.stack(responses_loc, 0)


# def calc_sizes(old_buffers, kernels, stride, trans):
#     k_h, k_w, _, _ = kernels.get_shape().as_list()
#     i_h, i_w = old_buffers[0][0][0].get_shape().as_list()
#     s_h, s_w = stride, stride
#     valid_input_h, valid_input_w = 1,1
#     if trans:
#         o_w = s_w * (i_w-1) + k_w
#         o_h = s_h * (i_h-1) + k_h
#     else:
#         valid_input_w = (i_w - k_w) + 1
#         valid_input_h = (i_h - k_h) + 1
#         o_w = int(np.ceil(i_w - (k_w-1))//s_w)
#         o_h = int(np.ceil(i_h - (k_h-1))//s_h)
#
#     return [valid_input_h, valid_input_w], [o_h, o_w]



def conv(prev_size, next_size, old_buffers, locations, kernels, biases, stride, window, trans, k, layer_name):
    """
    preform a convolution operation
    :param next_size: the next layer size, 2D array
    :param old_buffers: list of tuples (old buf_val, old buf_ind) in length of batch size
    :param kernels: filters to convolve with, 4D tensor in shape of HWIO
    :param biases: biases to add, 2D tensor (BATCH_SIZE, )
    :param stride: stride's size - [stride, stride]
    :param window: window's size - [window, window], used at max pool
    :param trans: False for regular convolution, True for transpose convolution
    :param k: next sparse size
    :return: new buffers
    """
    with tf.name_scope(layer_name):
        # prev_size, next_size = calc_sizes(old_buffers, kernels, stride, trans)
        # print(prev_size, next_size)

        batch_size = len(old_buffers)

        # create new buffers
        new_buffers = []
        new_buffers_size = next_size #if trans else prev_size
        for i in range(batch_size):
            buf_val = [tf.zeros(new_buffers_size)-1 for _ in range(k)]
            buf_ind = [tf.zeros(new_buffers_size, dtype=tf.int32) for _ in range(k)]
            # buf_loc = [tf.reshape(tf.range(new_buffers_size[0]*new_buffers_size[1], dtype=tf.int32),
            #                       new_buffers_size) for _ in range(k)]
            buf_loc = [tf.ones(new_buffers_size, dtype=tf.int32) for _ in range(k)]
            new_buffers.append((buf_val, buf_ind, buf_loc))

        # determine the input and output channels for conv operation
        c_i, c_o = kernels.shape[-2], kernels.shape[-1]
        if trans:
            c_i, c_o = c_o, c_i

        # convolve

        locations1 = None
        resp_loc = None

        for j in range(0, c_o, M_O):
            batch_image_size = next_size if trans else prev_size
            batch_resp_size = [batch_size] + batch_image_size + [min(M_O, c_o - j)]
            batch_responses = tf.zeros((batch_resp_size))

            for i in range(0, c_i, M_I):
                size = min(M_I, c_i.value-i)
                with tf.name_scope('W_%dT%d_%dT%d' % (j, j + M_O, i, i + size)):
                    # print('i', i, size)
                    # if trans:
                    #     old_buffers = [(old_buffers[0][0], old_buffers[0][1], locations)]
                    I, locations1 = reconstruct_batch(old_buffers, i, size)
                    if trans:
                        # print('convvv', locations1, locations, old_buffers[0][2])
                        I = unpool(I, locations1, next_size)
                        # print('I size', I)
                        conv_result = regular_conv(I, kernels[:, :, j:j + M_O, i:i + size], biases[j:j + M_O],
                                                   1, batch_size, trans, next_size)

                    else:
                        conv_result = regular_conv(I, kernels[:, :, i:i + size, j:j + M_O], biases[j:j + M_O],
                                                   1, batch_size, trans)

                    batch_responses += conv_result

            I = batch_responses

            # activation
            with tf.name_scope('acti_%dT%d' % (j, j+M_O)):
                resp_loc = tf.zeros(batch_resp_size, dtype=tf.int32)
                if not trans:
                    batch_responses = tf.nn.relu(batch_responses)
                    batch_responses, resp_loc = get_maxpool_argmax(batch_responses, window, stride)

            # update buffers
            with tf.name_scope('updBufs_%dT%d' % (j, j+M_O)):
                new_buffers = update_buffers(new_buffers, batch_responses, resp_loc, j, k)

        return new_buffers, locations, locations1, resp_loc


def copy_image_to_buffers(image):
    """
    Used only in the first layer, copy an image to buffers
    :param image: a 4D tensor (1,<img_shape>), assuming 1 image
    :return: buffers contain the image, each slice is a channel of the image
    """
    image_size = image.shape

    buf_val = []
    buf_ind = []
    buf_loc = []

    for i in range(image_size[-1]):
        buf_val.append(tf.cast(image[:, :, i], tf.float32))
        buf_ind.append(tf.constant(i, shape=image_size[:-1], dtype=tf.int32))
        buf_loc.append(tf.reshape(tf.range(image_size[0]*image_size[1], dtype=tf.int32), image_size[:-1]))

    return buf_val, buf_ind, buf_loc


def copy_batch_to_buffers(images, batch_size):
    """
    Copy batch of images to buffers
    :param images: 4D array (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    :return: list of buffers
    """
    # list of tuples of buffers, (buf_val, buf_ind, buf_loc) for each image
    buffers = []

    # batch_size = tf.shape(images)[0]
    for i in range(batch_size):
        image = images[i]
        buffers.append(copy_image_to_buffers(image))

    return buffers