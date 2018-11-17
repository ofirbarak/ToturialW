from AEconstants import *
from NewDeepNN2 import *
import matplotlib.pyplot as plt
from PIL import Image
import os, sys

# init kernels and biases
kernels = []
encbiases = []
decbiases = []
for i in range(NUM_LAYERS):
    kernels.append(tf.Variable(0.08 * tf.random_normal(KERNELS_SIZE[i]),name="kernels_"+str(i)))
    encbiases.append(tf.Variable(0.02 * tf.ones([KERNELS_SIZE[i][-1]]),name="encbiases_"+str(i)))
    decbiases.append(tf.Variable(0.02 * tf.ones([KERNELS_SIZE[i][-2]]),name="decbiases_"+str(i)))


def save_kernels(fil,fn):

    nfil = fil.shape[3]

    per_row = int(np.floor(np.sqrt(nfil)))
    per_col = int(np.ceil(nfil/per_row))

    for l in range(nfil):

        dims = np.shape(fil)
        nch = dims[2]

        im = np.zeros([(dims[0]+2)*per_row,(dims[1]+2)*per_col,nch])

        # global renormalization
        #fil = fil / np.max(np.abs(fil))
        #fil = fil * 127 + 128

        j = 0
        i = 0
        for k in range(nfil):

            f = np.reshape(fil[:,:,:,i*per_row+j],[dims[0],dims[1],nch])

            # local renormalization
            f = f / np.max(np.abs(f))
            f = f * 127 + 128

            im[j*(dims[0]+2)+1:j*(dims[0]+2)+dims[0]+1,i*(dims[1]+2)+1:i*(dims[1]+2)+dims[1]+1,:] = f

            j=j+1
            if j==per_row:
                i=i+1
                j=0

        if nch == 1:
            im = np.tile(im,(1,1,3))

        I = Image.fromarray(np.uint8(im))
        I = I.resize(((dims[1]+2)*per_row*8,(dims[0]+2)*per_col*8))
        I.save(fn + ".png")

    return im


def save_weights(sess,var_list,dn):
    if not os.path.exists(dn):
        os.makedirs(dn)

    for v in var_list:
        v_name = v.name[0:-2]
        v_name = v_name.replace("/","_")
        np.save(dn+'/'+v_name,sess.run(v))

    # print("\n>>> " + "Saving weights in " + dn)


def read_weights(sess,var_list,dn):
    if not os.path.exists(dn):
        print("dir " + dn + " doesn't exists (read_weights)")
        sys.exit()

    print("\n>>> Reading weights from " + dn)

    for v in var_list:
        v_name = v.name[0:-2]
        v_name = v_name.replace("/","_")
        nv = np.load(dn+'/'+v_name+".npy")

        v.load(nv, sess)

        print(v_name + '\t' + str(v.shape) + '\t' + str(np.prod(np.asarray(v.shape))))


def encoder(images, bs):
    # print('encoder')
    with tf.name_scope('copy_batch'):
        buffers = copy_batch_to_buffers(images, bs)

    trans = False
    locs = []

    buffers, loc, resp, I = conv(INPUT_SIZE, NLAYER_SIZES[0], buffers, None, kernels[0], encbiases[0],
                   STRIDE[0], WINDOW[0], trans, K[0], 'Enc' + str(0))
    locs.append(buffers[0][2])

    for i in range(1, NUM_LAYERS):
        buffers, loc, _, _ = conv(NLAYER_SIZES[i-1], NLAYER_SIZES[i], buffers, None, kernels[i], encbiases[i],
                       STRIDE[i], WINDOW[i], trans, K[i], 'Enc'+str(i))
        locs.append(buffers[0][2])


    return buffers, locs, resp, I


def decoder(foward_buffers):
    trans = True
    zbuffers, locs,_,_ = foward_buffers

    for i in range(NUM_LAYERS-1, 0, -1):
        zbuffers, _, _,_ = conv(NLAYER_SIZES[i], NLAYER_SIZES[i-1], zbuffers, locs[i], kernels[i], decbiases[i],
                        STRIDE[i], WINDOW[i], trans, K[i-1], 'Dec'+str(i))

        # replace buf_loc to forward buf_loc from encoder
        # for j in range(batch_size):
        #     zbuffers[j] = (zbuffers[j][0], zbuffers[j][1], foward_buffers[i-1][j][2])
    # print('decoder first layer')
    zbuffers, resp1, _, I = conv(NLAYER_SIZES[0], INPUT_SIZE, zbuffers, zbuffers[0][2], kernels[0], decbiases[0],
                   STRIDE[0], WINDOW[0], trans, INPUT_CH, 'Dec0')

    with tf.name_scope('reconstruct'):
        image, _ = reconstruct_batch(zbuffers, 0, INPUT_CH)
        return image, zbuffers, resp1, I


ae_input = tf.placeholder(tf.float32, [batch_size] + INPUT_SIZE + [INPUT_CH])
ae_output = decoder(encoder(ae_input, batch_size))[0]


rsp, rsp_crd = tf.nn.max_pool_with_argmax(ae_input, [1, 2, 2, 1],
                                             [1, 2, 2, 1], "VALID")
#rsp_crd = tf.ones_like(rsp_crd,tf.int32)*1
recon = tf.scatter_nd(tf.reshape(rsp_crd,(-1,1)), tf.reshape(rsp,(-1,)), (batch_size*226*226*1,))
recon = tf.reshape(recon,(batch_size,226,226,1))

def trainAE(images, lr=LEARNING_RATE, iterations=ITERATION):
    plt.figure(1)
    plt.ion()

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(ae_input - ae_output))
    tf.summary.scalar('mean_loss', tf.reduce_mean(loss))

    vlist = []

    if train_all_prev:
        for i in range(NUM_LAYERS):
            vlist.append(kernels[i])
            vlist.append(encbiases[i])
            vlist.append(decbiases[i])
    else:
        vlist.append(kernels[NUM_LAYERS-1])
        vlist.append(encbiases[NUM_LAYERS-1])
        vlist.append(decbiases[NUM_LAYERS-1])

    print("\nTraining:")
    for i in range(len(vlist)):
        print(vlist[i].name,end=' ')
        print(vlist[i].shape)
    print('\n')

    with tf.name_scope('train'):
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, var_list=vlist)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=vlist)


    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/test')
        init = tf.global_variables_initializer()

        sess.run(init)
        batch = images[:batch_size,:,:,:]

        if 0:

            print(batch[0,:3,:3,0])
            grsp, grsp_crd, grecon = sess.run([rsp, rsp_crd, recon], feed_dict={ae_input:  batch})
            print(grsp[0,:3,:3,0])
            print(grsp_crd[0,:3,:3,0])
            print(grecon[0,:6,:6,0])
            sys.exit()



        print("\n"+20*"-")
        # if load_all_prev:
        #     print("Reading ALL previus layers")
        #     read_weights(sess,kernels[0:NUM_LAYERS]+encbiases[0:NUM_LAYERS]+decbiases[0:NUM_LAYERS],"weights/")
        # else:
        #     if NUM_LAYERS>1:
        #         read_weights(sess,kernels[0:NUM_LAYERS-1]+encbiases[0:NUM_LAYERS-1]+decbiases[0:NUM_LAYERS-1],"weights/")
        #         print("reading a SINGLE previous layer")
        #     else:
        #         print("NOT reading weights from disk")

        losses = []
        for it in range(iterations):
            batch = images[np.random.choice(images.shape[0], batch_size)]

            summary, _, c = sess.run([merged, train_op, loss], feed_dict={ae_input: batch})
            train_writer.add_summary(summary, it)
           
            losses.append(c)

            if it % 30 == 0:
                # ################################
                # bufs, locs, resp, _ = sess.run(encoder(batch, batch_size))
                # l = locs[0]
                # image, zbufs, resp1, I = sess.run(decoder((bufs, locs, resp, _)))
                # print()
                # #
                # #
                # image, _ = sess.run(reconstruct_batch(bufs, 0, 4))
                # image = tf.convert_to_tensor(image)
                # image = sess.run(unpool(image, locs[0], INPUT_SIZE))
                # # # (nbufs, image2) = sess.run(conv(NLAYER_SIZES[0], NLAYER_SIZES[1], bufs, kernels[1], encbiases[1],
                # # #                                 STRIDE[1], WINDOW[1], True, 3, 'h'))
                # # # image = sess.run(reconstruct_batch(bufs, 0, 3))[0]
                # # # image2 = sess.run(reconstruct_batch(bufs1, 0, 1))
                # # # print(images[0,:,:,0])
                # # # print(resp)
                # # # image = sess.run(unpool(reconstruct_batch(bufs, 0, 3), bufs[-1], INPUT_SIZE))
                # print(image.shape)
                # # plt.subplot(121)
                # #
                # image[image > 1] = 1
                # image[image < 0] = 0
                # # image = (image - np.min(image))
                # # image = image / np.max(image)
                # plt.imshow(image[0, :, :, 0], cmap='gray')
                # #
                # # plt.subplot(122)
                # # # image2[image2 > 1] = 1
                # # # image2[image2 < 0] = 0
                # # image2 = (image2 - np.min(image2))
                # # image2 = image2 / np.max(image2)
                # #
                # # plt.imshow(image2[0, :, :, 0], cmap='gray')
                # plt.show()
                # plt.pause(5)
                # # ###################################

                save_weights(sess,kernels[0:NUM_LAYERS]+encbiases[0:NUM_LAYERS]+decbiases[0:NUM_LAYERS],"weights")
                kers = save_kernels(sess.run(kernels[0]),"kernels/kers")

                plt.subplot(221)
                plt.plot(losses)

                image = batch[0].copy()
                image = (image - np.min(image))
                image = image / np.max(image)

                aeimage = sess.run(ae_output, feed_dict={ae_input: batch})
                aeimage = aeimage[0]

                print(np.min(aeimage))
                print(np.max(aeimage))

                # aeimage = (aeimage - np.min(aeimage))
                # aeimage = aeimage / np.max(aeimage)

                aeimage[aeimage>1]=1
                aeimage[aeimage<0]=0

                plt.subplot(222)
                plt.imshow(kers/255)
                plt.subplot(223)
                if GRAY:
                    plt.imshow(image[:,:,0], cmap='gray')
                else:
                    plt.imshow(image)
                plt.subplot(224)
                if GRAY:
                    plt.imshow(aeimage[:,:,0], cmap='gray')
                else:
                    plt.imshow(aeimage)
                plt.savefig("fig.png",dpi=300)
                plt.show()
                plt.draw()
                plt.pause(0.0001)
