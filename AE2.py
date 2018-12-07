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
    kernels.append(tf.Variable(0.1 * tf.random_normal(KERNELS_SIZE[i]),name="kernels_"+str(i)))
    encbiases.append(tf.Variable(0.0 * tf.ones([KERNELS_SIZE[i][-1]]),name="encbiases_"+str(i)))
    decbiases.append(tf.Variable(0.0 * tf.ones([KERNELS_SIZE[i][-2]]),name="decbiases_"+str(i)))

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

def proc_image(I):
    if WHITEN:
        for c in range(I.shape[2]):
            ch = I[:,:,c]
            ch = ch / (np.max(ch)-np.min(ch)) + 0.5
    else:
        I[I < 0] = 0
        I[I > 1] = 1

    return I

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

def sparse(x,k=1,axis=3):
    # return x
    #return tf.multiply(x,tf.nn.softmax(13*x,axis))
    
    z = tf.multiply(x,tf.one_hot(tf.argmax(x,axis),depth=tf.shape(x)[axis],axis=axis))

    x = x - z
    
    for i in range(k-1):
        tz = tf.multiply(x,tf.one_hot(tf.argmax(x,axis),depth=tf.shape(x)[axis],axis=axis))
        z = z + tz
        x = x - tz

    return z

def naive_encoder(I):

    for i in range(NUM_LAYERS):
        I = tf.nn.conv2d(I, kernels[i], [1, STRIDE[i], STRIDE[i], 1], 'SAME')
        I = tf.nn.bias_add(I, encbiases[i])
        I = tf.nn.relu(I)
        I = sparse(I,Ksp[i])
        
    return I

def naive_decoder(I):

    bs = tf.shape(I)[0]
    sz = tf.shape(I)[1]

    for i in range(NUM_LAYERS-1,-1,-1):
        sz = sz * STRIDE[i]
        print("--"*20)
        print(I.shape)
        print(kernels[i].shape)
        print(NUN_KER_comp[i])
        print(STRIDE[i])
        I = tf.nn.conv2d_transpose(I, kernels[i], [bs]+[sz,sz]+[NUN_KER_comp[i]], [1, STRIDE[i], STRIDE[i], 1], 'SAME')
        I = tf.nn.bias_add(I, decbiases[i])
        if i>0:
            I = tf.nn.relu(I)
            I = sparse(I,Ksp[i-1])
        print(I.shape)
        print("=="*20)
        
    return I




def encoder(images, bs):
    # print('encoder')
    with tf.name_scope('copy_batch'):
        buffers = copy_batch_to_buffers(images, bs)

    trans = False
    locations = []

    buffers, _, _, loc = conv(INPUT_SIZE, NLAYER_SIZES[0], buffers, None, kernels[0], encbiases[0],
                   STRIDE[0], WINDOW[0], trans, Ksp[0], 'Enc' + str(0))
    locations.append(loc)
    # loc = buffers[0][2]
    # buf_locs = [buffers[layer][2] for layer in range(batch_size)]
    # print('endcoder second layer')
    for i in range(1, NUM_LAYERS):
        buffers, _, _, loc = conv(NLAYER_SIZES[i-1], NLAYER_SIZES[i], buffers, None, kernels[i], encbiases[i],
                       STRIDE[i], WINDOW[i], trans, Ksp[i], 'Enc'+str(i))
        locations.append(loc)
        # buf1_locs = [buffers1[layer][2] for layer in range(batch_size)]

    # print('lennnnn', len(buffers))
    return buffers, locations, [buffers[0][2], buffers[0][2]]


def decoder(zbuffers):
    # print('decoder')
    zbuffers, locations, bufs = zbuffers
    trans = True

    # zbufs = []
    x1,x2 = None ,None
    zbuffers1 = None
    for i in range(NUM_LAYERS-1, 0, -1):
        zbuffers, x1,x2, loc1 = conv(NLAYER_SIZES[i], NLAYER_SIZES[i-1], zbuffers, locations[i], kernels[i], decbiases[i],
                        STRIDE[i], WINDOW[i], trans, Ksp[i-1], 'Dec'+str(i))

        # print(locations[0])
        for layer in range(batch_size):
            zbuffers[layer] = (zbuffers[layer][0], zbuffers[layer][1], tf.expand_dims(locations[i-1][layer], 0))
        # zbufs.append([zbuffers1[0][0], zbuffers1[0][1], locations[0]])
        # print('decoder', locations[0])
        # zbuffers[0] = zbuffers[0][0], zbuffers[0][1], locations[0]

    # print('decoder first layer')
    zbuffers, l, l1, _ = conv(NLAYER_SIZES[0], INPUT_SIZE, zbuffers, locations[0], kernels[0], decbiases[0],
                   STRIDE[0], WINDOW[0], trans, INPUT_CH, 'Dec0', False)

    with tf.name_scope('reconstruct'):
        image, _ = reconstruct_batch(zbuffers, 0, INPUT_CH)
        return image, l, l1, l, l1


ae_input = tf.placeholder(tf.float32, [batch_size] + INPUT_SIZE + [INPUT_CH])
#ae_output = decoder(encoder(ae_input, batch_size))[0]
ae_output = naive_decoder(naive_encoder(ae_input))


# rsp, rsp_crd = tf.nn.max_pool_with_argmax(ae_input, [1, 2, 2, 1],
#                                              [1, 2, 2, 1], "VALID")
# #rsp_crd = tf.ones_like(rsp_crd,tf.int32)*1
# recon = tf.scatter_nd(tf.reshape(rsp_crd,(-1,1)), tf.reshape(rsp,(-1,)), (batch_size*226*226*1,))
# recon = tf.reshape(recon,(batch_size,226,226,1))


# visualization
ks = KER_DIM[1]*STRIDE[0]*STRIDE[1]
ks = 6*8

print(NUN_KER[1]*ks*ks*INPUT_CH)
vis_stm = tf.Variable(0.04 * tf.random_normal([NUN_KER[1],ks,ks,INPUT_CH]),name="ker_vis")
ks1 = int(ks / STRIDE[0])
ks1 = int(ks1 / STRIDE[1])

print(NUN_KER[1]*ks1*ks1*NUN_KER[1])
vis_obj = tf.placeholder(tf.float32, shape=[NUN_KER[1],ks1,ks1,NUN_KER[1]])
np_vis_obj = np.zeros([NUN_KER[1],ks1,ks1,NUN_KER[1]])
for i in range(NUN_KER[1]):
    np_vis_obj[i,int(ks1/2),int(ks1/2),i] = 59

if NUM_LAYERS > 1:
    recon = naive_decoder(vis_obj)

def trainAE(images, lr=LEARNING_RATE, iterations=ITERATION):
    plt.figure(1)
    plt.ion()

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(ae_input - ae_output))
    # tf.summary.scalar('mean_loss', tf.reduce_mean(loss))

    if NUM_LAYERS > 1:
        vis_loss = tf.reduce_mean(tf.square(vis_obj - naive_encoder(vis_stm)))
        train_vis = tf.train.AdamOptimizer(learning_rate=0.001).minimize(vis_loss, var_list=vis_stm)

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
        #train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, var_list=vlist)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=vlist)


    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/test')
        init = tf.global_variables_initializer()

        sess.run(init)
        
        test_batch = images[np.random.choice(images.shape[0], batch_size)]

        print("\n"+20*"-")
        if load_all_prev:
            print("Reading ALL previous layers")
            read_weights(sess,kernels[0:NUM_LAYERS]+encbiases[0:NUM_LAYERS]+decbiases[0:NUM_LAYERS],"weights/")
        else:
            if NUM_LAYERS>1:
                read_weights(sess,kernels[0:NUM_LAYERS-1]+encbiases[0:NUM_LAYERS-1]+decbiases[0:NUM_LAYERS-1],"weights/")
                print("reading a SINGLE previous layer")
            else:
                print("NOT reading weights from disk")


        losses = []
        for it in range(iterations):
            batch = images[np.random.choice(images.shape[0], batch_size)]


            _, c = sess.run([train_op, loss], feed_dict={ae_input: batch})
            # summary, _, c = sess.run([merged, train_op, loss], feed_dict={ae_input: batch})
            # train_writer.add_summary(summary, it)

            c = sess.run(loss, feed_dict={ae_input: test_batch})
            losses.append(c)
         
            if it % 500 == 0:

                save_weights(sess,kernels[0:NUM_LAYERS]+encbiases[0:NUM_LAYERS]+decbiases[0:NUM_LAYERS],"weights")

                plt.clf()
                plt.subplot(231)
                if len(losses) > 100:
                    plt.plot(losses[100:])
                else:
                    plt.plot(losses)
                    
                image = test_batch[0].copy()

                aeimage = sess.run(ae_output, feed_dict={ae_input: test_batch})
                aeimage = aeimage[0]
                
                aeimage = proc_image(aeimage)
                image = proc_image(image)

                kers = save_kernels(sess.run(kernels[0]),"kernels/kers")
                plt.subplot(232)
                plt.imshow(kers/255)

                if NUM_LAYERS > 1:
                    
                    for k in range(100):
                        tv = sess.run(train_vis, feed_dict={vis_obj: np_vis_obj})
                    print("opt err")
                    print(sess.run(vis_loss, feed_dict={vis_obj: np_vis_obj}))
                    kers2 = sess.run(vis_stm)
                    kers3 = sess.run(recon, feed_dict={vis_obj: np_vis_obj})
                    print("mean abs inv.")
                    print(np.mean(np.abs(kers3)))

                    kers2 = np.transpose(kers2,(1,2,3,0))
                    kers3 = np.transpose(kers3,(1,2,3,0))

                    kers2 = save_kernels(kers2,"kernels/kers2")
                    kers3 = save_kernels(kers3,"kernels/kers3")
                    plt.subplot(233)
                    plt.imshow(kers2/255)
                    plt.title("opt.")
                    plt.subplot(236)
                    plt.imshow(kers3/255)
                    plt.title("inv.")
                
                plt.subplot(234)
                if GRAY:
                    plt.imshow(image[:,:,0], cmap='gray')
                else:
                    plt.imshow(image)
                plt.subplot(235)
                if GRAY:
                    plt.imshow(aeimage[:,:,0], cmap='gray')
                else:
                    plt.imshow(aeimage)
                plt.savefig("fig.png",dpi=400)
                plt.show()
                plt.draw()
                plt.pause(0.0001)
