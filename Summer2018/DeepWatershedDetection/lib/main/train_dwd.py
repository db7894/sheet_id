
import os
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/utils/')
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/models/')
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/datasets/')
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/main/')
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/roi_data_layer/')
from main.config import cfg

from models.dwd_net import build_dwd_net

from factory import get_imdb
from contrib import slim
from safe_softmax_wrapper import safe_softmax_cross_entropy_with_logits
import roidb as rdl_roidb
from layer import RoIDataLayer
from prefetch_wrapper import PrefetchWrapper


from datasets.fcn_groundtruth import stamp_class, stamp_directions, stamp_energy, stamp_bbox,\
    try_all_assign,get_gt_visuals,get_map_visuals

nr_classes = None
# - make different FCN architecture available --> RefineNet, DeepLabv3, standard fcn

def main(parsed):
    args = parsed[0]
    print(args)
    iteration = 1

    np.random.seed(cfg.RNG_SEED)

    # load database
    imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val = load_database(args)

    global nr_classes
    nr_classes = len(imdb._classes)
    args.nr_classes.append(nr_classes)

    # replaces keywords with function handles in training assignements
    save_objectness_function_handles(args,imdb)


    # Debug stuffs
    #
    # try_all_assign(data_layer,args,100)
    # dws_list = perform_dws(data["dws_energy"], data["class_map"], data["bbox_fcn"])
    #

    # def show_image(data, gt_boxes=None):
    #     from PIL import Image, ImageDraw
    #     if gt_boxes is None:
    #         im = Image.fromarray(data[0].astype("uint8"))
    #         im.show()
    #     else:
    #         im = Image.fromarray(data[0].astype("uint8"))
    #         draw = ImageDraw.Draw(im)
    #         # overlay GT boxes
    #         for row in gt_boxes[0]:
    #             draw.rectangle(((row[0], row[1]), (row[2], row[3])), fill="red")
    #         im.show()
    #     return
    #
    # fetched_dat = data_layer.forward(args, [args.training_assignements[0]], None)

    # show_image(fetched_dat["data"],fetched_dat["gt_boxes"])
    # show_image(fetched_dat["data"], None)


    # tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # input and output tensors
    if "DeepScores" in args.dataset:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"

    elif "MUSICMA" in args.dataset:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"

    else:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        resnet_dir = cfg.PRETRAINED_DIR+"/ImageNet/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/VOC2012/"


    if not (len(args.training_help) == 1 and args.training_help[0] is None):
        # initialize helper_input
        helper_input  = tf.placeholder(tf.float32, shape=[None, None, None, input.shape[-1]+1])
        feed_head = slim.conv2d(helper_input, input.shape[-1], [3, 3], scope='gt_feed_head')
        input = feed_head

    print("Initializing Model:" + args.model)
    # model has all possible output heads (even if unused) to ensure saving and loading goes smoothly
    network_heads, init_fn = build_dwd_net(
        input, model=args.model,num_classes=nr_classes, pretrained_dir=resnet_dir, substract_mean=False)



    # initialize tasks
    preped_assign = []
    for assign in args.training_assignements:
        [loss, optim, gt_placeholders, scalar_summary_op,images_summary_op, images_placeholders, mask_placholders] = initialize_assignement(assign,imdb,network_heads,sess,data_layer,input,args)
        preped_assign.append([loss, optim, gt_placeholders, scalar_summary_op,images_summary_op, images_placeholders, mask_placholders])



    # init tensorflow session
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    # load model weights
    checkpoint_dir = get_checkpoint_dir(args)
    checkpoint_name =  "backbone"
    if args.continue_training == "True":
        print("Loading checkpoint")
        saver.restore(sess, checkpoint_dir + "/" + checkpoint_name)
    elif args.pretrain_lvl == "deepscores_to_musicma":
        pretrained_vars = []
        for var in slim.get_model_variables():
            if not ("class_pred" in var.name):
                pretrained_vars.append(var)
        print("Loading network pretrained on Deepscores for Muscima")
        loading_checkpoint_name = cfg.PRETRAINED_DIR+"/DeepScores_to_Muscima/" + "backbone"
        init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
        init_fn(sess)

    else:
        if args.pretrain_lvl == "semseg":
            #load all variables except the ones in scope "deep_watershed"
            pretrained_vars = []
            for var in slim.get_model_variables():
                if not("deep_watershed" in var.name or "gt_feed_head" in var.name):
                    pretrained_vars.append(var)

            print("Loading network pretrained on semantic segmentation")
            loading_checkpoint_name = refinenet_dir + args.model + ".ckpt"
            init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
            init_fn(sess)
        elif args.pretrain_lvl == "class":
            print("Loading pretrained weights for level: " + args.pretrain_lvl)
            init_fn(sess)
        else:
            print("Not loading a pretrained network")

    # set up tensorboard
    writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

    # execute tasks
    for do_a in args.do_assign:
        assign_nr = do_a["assign"]
        do_itr = do_a["Itrs"]
        training_help = args.training_help[do_a["help"]]
        iteration = execute_assign(args,input,saver, sess, checkpoint_dir, checkpoint_name, data_layer, writer, network_heads,
                                   do_itr,args.training_assignements[assign_nr],preped_assign[assign_nr],iteration,training_help)


    # execute combined tasks
    for do_comb_a in args.combined_assignements:
        #iteration = execute_combined_assign(do_comb_a)
        do_comb_itr = do_comb_a["Itrs"]
        rm_length = do_comb_a["Running_Mean_Length"]
        loss_factors = do_comb_a["loss_factors"]
        orig_assign = [args.training_assignements[i] for i in do_comb_a["assigns"]]
        preped_assigns = [preped_assign[i] for i in do_comb_a["assigns"]]
        training_help = None # unused atm
        execute_combined_assign(args,data_layer,training_help,orig_assign,preped_assigns,loss_factors,do_comb_itr,iteration,input,rm_length,
                            network_heads,sess,checkpoint_dir,checkpoint_name,saver,writer)

    print("done :)")

    # traind on combined assigns
    # for comb_assign in args.combined_assignements:
    #     train_on_comb_assignment()


def execute_combined_assign(args,data_layer,training_help,orig_assign,preped_assigns,loss_factors,do_comb_itr,iteration,input_ph,rm_length,
                            network_heads,sess,checkpoint_dir,checkpoint_name,saver,writer):

    # init data layer
    if args.prefetch == "True":
        data_layer = PrefetchWrapper(data_layer.forward, args.prefetch_len, args, orig_assign, training_help)

    # combine losses
    past_losses = np.ones((len(loss_factors), rm_length), np.float32)
    loss_scalings_placeholder = tf.placeholder(tf.float32,[len(loss_factors)])
    loss_tot = None
    for i in range(len(preped_assigns)):
        if loss_tot is None:
            loss_tot = preped_assigns[i][0]*loss_scalings_placeholder[i]
        else:
            loss_tot += preped_assigns[i][0] * loss_scalings_placeholder[i]

    # init optimizer
    with tf.variable_scope("combined_opt"+str(0)):
        var_list = [var for var in tf.trainable_variables()]
        optim = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=0.995).minimize(loss_tot, var_list=var_list)
    opt_inizializers = [var.initializer for var in tf.global_variables() if "combined_opt"+str(0) in var.name]
    sess.run(opt_inizializers)
    # compute step
    print("training on combined assignments")
    print("for " + str(do_comb_itr) + " iterations")

    # waste elements off queue because queue clear does not work
    for i in range(14):
        data_layer.forward(args, orig_assign, training_help)


    for itr in range(iteration, (iteration + do_comb_itr)):
        # load batch - only use batches with content
        batch_not_loaded = True
        while batch_not_loaded:
            blob = data_layer.forward(args, orig_assign, training_help)
            batch_not_loaded = len(blob["gt_boxes"].shape) != 3 or  sum(["assign" in key for key in  blob .keys()]) != len(preped_assigns)


        if blob["helper"] is not None:
            input_data = np.concatenate([blob["data"], blob["helper"]], -1)
            feed_dict = {input_ph: input_data}
        else:
            # pad input with zeros
            # input_data = np.concatenate([blob["data"]*0, blob["data"]*0], -1)
            # feed_dict = {input: blob["data"], helper_input: input_data}
            if len(args.training_help) == 1:
                feed_dict = {input_ph: blob["data"]}
            else:
                # pad input with zeros
                input_data = np.concatenate([blob["data"], blob["data"] * 0], -1)
                feed_dict = {input_ph: input_data}


        for i1 in range(len(preped_assigns)):
            gt_placeholders = preped_assigns[i1][2]
            mask_placeholders = preped_assigns[i1][6]
            for i2 in range(len(gt_placeholders)):
                # only one assign
                feed_dict[gt_placeholders[i2]] = blob["assign"+str(i1)]["gt_map" + str(len(gt_placeholders) - i2 - 1)]
                feed_dict[mask_placeholders[i2]] = blob["assign" + str(i1)]["mask" + str(len(gt_placeholders) - i2 - 1)]

        # compute running mean for losses
        feed_dict[loss_scalings_placeholder] = loss_factors/np.maximum(np.mean(past_losses,1),[1.0E-6,1.0E-6,1.0E-6])

        # train step
        fetch_list = list()
        fetch_list.append(optim)
        fetch_list.append(loss_tot)
        for preped_a in preped_assigns:
            fetch_list.append(preped_a[0])
        fetches = sess.run(fetch_list, feed_dict=feed_dict)

        past_losses[:,:-1] = past_losses[:,1:] # move by one timestep
        past_losses[:,-1] = fetches[-3:] # add latest loss

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(fetches[1])
            print(past_losses)


        if itr % args.tensorboard_interval == 0 or itr == 1:
            for i in range(len(preped_assigns)):
                _, _, _, scalar_summary_op, images_summary_op, images_placeholders, _ = preped_assigns[i]
                post_assign_to_tensorboard(orig_assign[i],i,scalar_summary_op,network_heads,feed_dict,itr,sess,writer,blob,images_placeholders,images_summary_op)

        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)


    iteration = (iteration + do_comb_itr)
    if args.prefetch == "True":
        data_layer.kill()

    return iteration


def post_assign_to_tensorboard(assign,assign_nr,scalar_summary_op,network_heads,feed_dict,itr,sess,writer,blob,images_placeholders,images_summary_op):
    fetch_list = [scalar_summary_op]
    # fetch sub_predicitons
    nr_feature_maps = len(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])

    [fetch_list.append(
        network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps - (x + 1)]) for x
        in
        range(len(assign["ds_factors"]))]

    summary = sess.run(fetch_list, feed_dict=feed_dict)
    writer.add_summary(summary[0], float(itr))

    # use predicted feature maps
    # TODO predict boxes

    # debug logits
    # if itr ==1:
    #     hist_ph = tf.placeholder(tf.uint8, shape=[1, summary[1].shape[3]])
    #     logits_sum = tf.summary.histogram('logits_means', hist_ph)
    #
    # h_sum = sess.run([logits_sum], feed_dict={hist_ph: np.mean(summary[1],(1,2))})
    # writer.add_summary(h_sum[0], float(itr))


    gt_visuals = get_gt_visuals(blob, assign, assign_nr, pred_boxes=None, show=False)
    map_visuals = get_map_visuals(summary[1:], assign, show=False)
    images_feed_dict = get_images_feed_dict(assign, blob, gt_visuals, map_visuals, images_placeholders)
    # save images to tensorboard
    summary = sess.run([images_summary_op], feed_dict=images_feed_dict)
    writer.add_summary(summary[0], float(itr))
    return None



def initialize_assignement(assign,imdb,network_heads,sess,data_layer,input,args):
    gt_placeholders = get_gt_placeholders(assign,imdb)

    loss_mask_placeholders = [tf.placeholder(tf.float32, shape=[None, None,1]) for x in assign["ds_factors"]]

    debug_fetch = dict()

    if assign["stamp_func"][0] == "stamp_directions":
        loss_components = []
        for x in range(len(assign["ds_factors"])):
            debug_fetch[str(x)] = dict()
            # # mask, where gt is zero
            split1, split2 = tf.split(gt_placeholders[x],2,-1)
            debug_fetch[str(x)]["split1"] = split1

            mask = tf.squeeze(split1 > 0, -1)
            debug_fetch[str(x)]["mask"] = mask

            masked_pred = tf.boolean_mask(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x], mask)
            debug_fetch[str(x)]["masked_pred"]= masked_pred

            masked_gt = tf.boolean_mask(gt_placeholders[x], mask)
            debug_fetch[str(x)]["masked_gt"] = masked_gt

            # norm prediction
            norms = tf.norm(masked_pred,ord="euclidean",axis=-1,keep_dims=True)
            masked_pred = masked_pred/norms
            debug_fetch[str(x)]["masked_pred_normed"] = masked_pred

            # inner product
            # inner = tf.diag_part(tf.tensordot(masked_gt,tf.transpose(masked_pred),1))
            # debug_fetch[str(x)]["inner"] = inner

            gt_1, gt_2 = tf.split(masked_gt, 2, -1)
            pred_1, pred_2 = tf.split(masked_pred, 2, -1)
            inner_2 = gt_1*pred_1+gt_2*pred_2
            debug_fetch[str(x)]["inner_2"] = inner_2
            # round to 4 digits after dot
            # multiplier = tf.constant(10 ** 4, dtype=tf.float32)
            # inner_rounded = tf.round(inner_2 * multiplier) / multiplier
            # debug_fetch[str(x)]["inner_rounded"] = inner_rounded

            # cap to [-1,1] due to numerical instability

            inner_2 = tf.maximum(tf.constant(-1, dtype=tf.float32),tf.minimum(tf.constant(1, dtype=tf.float32),inner_2))

            acos_inner = tf.acos(inner_2)
            debug_fetch[str(x)]["acos_inner"] = acos_inner

            loss_components.append(acos_inner)
    else:
        nr_feature_maps = len(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])
        nr_ds_factors = len(assign["ds_factors"])
        if assign["stamp_args"]["loss"] == "softmax":
            # loss_components = [safe_softmax_cross_entropy_with_logits(logits=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps-nr_ds_factors+x],
            #                                                 labels=gt_placeholders[x]) for x in range(nr_ds_factors)]
            loss_components = [tf.nn.softmax_cross_entropy_with_logits(logits=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps-nr_ds_factors+x],
                                                            labels=gt_placeholders[x], dim=-1) for x in range(nr_ds_factors)]
            debug_fetch["loss_components_softmax"] = loss_components
        else:
            loss_components = [tf.losses.mean_squared_error(
                predictions=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps-nr_ds_factors+x],
                labels=gt_placeholders[x],reduction="none") for x in range(nr_ds_factors)]
            debug_fetch["loss_components_mse"] = loss_components

    comp_multy = []
    for i in range(len(loss_components)):
        # maybe expand dims
        if len(loss_components[i][0].shape)==2:
            cond_result = tf.expand_dims(loss_components[i][0], -1)
        else:
            cond_result = loss_components[i][0]
        comp_multy.append(tf.multiply(cond_result, loss_mask_placeholders[i]))
        # debug_fetch["loss_components_multy"+str(i)] = tf.multiply(loss_components[i], loss_mask_placeholders[i])
        # debug_fetch["loss_components_multyaaaa" + str(i)] = tf.multiply(loss_components[i], 3)
    # call tf.reduce mean on each loss component
    final_loss_components = [tf.reduce_mean(x) for x in comp_multy]

    #################################################################################################################
    # debug  losses
    # load batch - only use batches with content
    # batch_not_loaded = True
    # while batch_not_loaded:
    #
    #     blob = data_layer.forward(args, [assign], None)
    #     batch_not_loaded = len(blob["gt_boxes"].shape) != 3
    #
    #     feed_dict = {input: blob["data"]}
    #     for i in range(len(gt_placeholders)):
    #         feed_dict[gt_placeholders[i]] = blob["assign0"]["gt_map" + str(len(gt_placeholders)-i-1)]
    #         feed_dict[loss_mask_placeholders[i]] = blob["assign0"]["mask" + str(len(gt_placeholders) - i - 1)]
    #
    # sess.run(tf.global_variables_initializer())
    # 1==1
    # # train softmax
    # loss_fetch = sess.run([final_loss_components,comp_multy,
    #                        loss_components], feed_dict=feed_dict)
    # loss_comp_final, comp_mul, loss_mse = loss_fetch
    #
    # loss_fetch = sess.run([loss_components], feed_dict=feed_dict)


    # from PIL import Image
    # Image.fromarray(np.squeeze(blob["assign0"]["mask" + str(len(gt_placeholders) - i - 1)],-1).astype(np.uint8)).show()
    # Image.fromarray(np.squeeze(blob["data"],-1).astype(np.uint8)[0]).show()
    # loss_fetch  = np.squeeze((loss_fetch[0][0]/np.amax(loss_fetch[0][0])*255).astype(np.uint8),-1)
    # Image.fromarray(loss_fetch).show()
    #
    #
    # feature_maps_fetch = sess.run(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]], feed_dict=feed_dict)
    # for map in feature_maps_fetch:
    #     print(map.shape)
    #
    # # train step
    #
    # [split] = sess.run([debug_fetch[str(x)]["split1"]], feed_dict=feed_dict)
    # [pred] = sess.run([network_heads[assign["stamp_func"][0]][x]], feed_dict=feed_dict)
    # [mask] = sess.run([debug_fetch[str(x)]["mask"]], feed_dict=feed_dict)
    # [masked_gt] = sess.run([debug_fetch[str(x)]["masked_gt"]], feed_dict=feed_dict)
    # [masked_pred_normed] = sess.run([debug_fetch[str(x)]["masked_pred_normed"]], feed_dict=feed_dict)
    # [inner] = sess.run([debug_fetch[str(x)]["inner"]], feed_dict=feed_dict)
    # [inner_2] = sess.run([debug_fetch[str(x)]["inner_2"]], feed_dict=feed_dict)
    # [inner_rounded] = sess.run([debug_fetch[str(x)]["inner_rounded"]], feed_dict=feed_dict)
    # debug_fetch[str(x)].keys()
    # #################################################################################################################

    # TODO replace by weight mask
    # # potentially mask out zeros
    # if assign["mask_zeros"]:
    #     # only compute loss where GT is not zero intended for "directional donuts"
    #     masked_components = []
    #     for x in range(len(assign["ds_factors"])):
    #         mask = tf.squeeze(gt_placeholders[x] > 0, -1)
    #         masked_components.append(tf.boolean_mask(loss_components[x], mask))
    #
    #     loss_components = masked_components



    # replace with loss of last layer if is nan (can happen if last mask has no directions on it)
    #loss_components = [tf.cond(tf.is_nan(x), lambda: loss_components[len(loss_components)-1], lambda: x) for x in loss_components]

    stacked_components = tf.stack(final_loss_components)


    if assign["layer_loss_aggregate"] == "min":
        loss = tf.reduce_min(stacked_components)
    elif assign["layer_loss_aggregate"] == "avg":
        loss = tf.reduce_mean(stacked_components)
    else:
        raise NotImplementedError("unknown layer aggregate")


    # init optimizer
    var_list = [var for var in tf.trainable_variables()]
    optim = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=0.995).minimize(loss, var_list=var_list)


    # init summary operations
    # define summary ops
    scalar_sums = []

    scalar_sums.append(tf.summary.scalar("loss " + get_config_id(assign)+":", loss))

    for comp_nr in range(len(loss_components)):
        scalar_sums.append(tf.summary.scalar("loss_component " + get_config_id(assign) + "Nr"+str(comp_nr)+":", final_loss_components[comp_nr]))

    scalar_summary_op = tf.summary.merge(scalar_sums)



    images_sums = []
    images_placeholders = []

    # feature maps
    for i in range(len(assign["ds_factors"])):
        sub_prediction_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        images_placeholders.append(sub_prediction_placeholder)
        images_sums.append(tf.summary.image('sub_prediction_' + str(i)+"_"+get_config_id(assign), sub_prediction_placeholder))


    helper_img = tf.placeholder(tf.uint8, shape=[1, None, None, 1])
    images_placeholders.append(helper_img)
    images_sums.append(tf.summary.image('helper' + str(i)+get_config_id(assign), helper_img))


    final_pred_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
    images_placeholders.append(final_pred_placeholder)
    images_sums.append(tf.summary.image('final_predictions_' + str(i)+get_config_id(assign), final_pred_placeholder))
    images_summary_op = tf.summary.merge(images_sums)

    return loss, optim, gt_placeholders, scalar_summary_op,images_summary_op, images_placeholders, loss_mask_placeholders





def execute_assign(args,input,saver, sess, checkpoint_dir, checkpoint_name, data_layer, writer, network_heads,
                   do_itr, assign, prepped_assign, iteration,training_help):
    loss, optim, gt_placeholders, scalar_summary_op, images_summary_op, images_placeholders, mask_placeholders = prepped_assign

    if args.prefetch == "True":
        data_layer = PrefetchWrapper(data_layer.forward, args.prefetch_len, args, [assign], training_help)



    print("training on:" + str(assign))
    print("for " + str(do_itr)+ " iterations")
    for itr in range(iteration, (iteration + do_itr)):
        # load batch - only use batches with content
        batch_not_loaded = True
        while batch_not_loaded:
            blob = data_layer.forward(args, [assign], training_help)
            if int(gt_placeholders[0].shape[-1]) != blob["assign0"]["gt_map0"].shape[-1] or len(blob["gt_boxes"].shape) != 3:
                print("skipping queue element")
            else:
                batch_not_loaded = False


        if blob["helper"] is not None:
            input_data = np.concatenate([blob["data"],blob["helper"]],-1)
            feed_dict = {input: input_data}
        else:
            # pad input with zeros
            # input_data = np.concatenate([blob["data"]*0, blob["data"]*0], -1)
            # feed_dict = {input: blob["data"], helper_input: input_data}
            if len(args.training_help) == 1:
                feed_dict = {input: blob["data"]}
            else:
                # pad input with zeros
                input_data = np.concatenate([blob["data"], blob["data"]*0], -1)
                feed_dict = {input: input_data}

        for i in range(len(gt_placeholders)):
            # only one assign
            feed_dict[gt_placeholders[i]] = blob["assign0"]["gt_map" + str(len(gt_placeholders)-i-1)]
            feed_dict[mask_placeholders[i]] = blob["assign0"]["mask" + str(len(gt_placeholders) - i - 1)]


        # train step
        _, loss_fetch = sess.run([optim, loss], feed_dict=feed_dict)

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(loss_fetch)

        if itr % args.tensorboard_interval == 0 or itr == 1:
            fetch_list = [scalar_summary_op]
            # fetch sub_predicitons
            nr_feature_maps = len(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])

            [fetch_list.append(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps-(x+1)]) for x in
             range(len(assign["ds_factors"]))]

            summary = sess.run(fetch_list, feed_dict=feed_dict)
            writer.add_summary(summary[0], float(itr))

            # use predicted feature maps
            # TODO predict boxes

            # debug logits
            # if itr ==1:
            #     hist_ph = tf.placeholder(tf.uint8, shape=[1, summary[1].shape[3]])
            #     logits_sum = tf.summary.histogram('logits_means', hist_ph)
            #
            # h_sum = sess.run([logits_sum], feed_dict={hist_ph: np.mean(summary[1],(1,2))})
            # writer.add_summary(h_sum[0], float(itr))


            gt_visuals = get_gt_visuals(blob, assign, 0, pred_boxes=None, show=False)
            map_visuals = get_map_visuals(summary[1:], assign, show=False)
            images_feed_dict = get_images_feed_dict(assign, blob, gt_visuals, map_visuals, images_placeholders)
            # save images to tensorboard
            summary = sess.run([images_summary_op], feed_dict=images_feed_dict)
            writer.add_summary(summary[0], float(itr))

        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)

    iteration = (iteration + do_itr)
    if args.prefetch == "True":
        data_layer.kill()

    return iteration





def get_images_feed_dict(assign,blob,gt_visuals,map_visuals,images_placeholders):
    feed_dict = dict()
    # for i in range(len(assign["ds_factors"])*2):
    #     if i%2 ==0:
    #         # prediction
    #         feed_dict[images_placeholders[i]] = map_visuals[i/2]
    #
    #     else:
    #         feed_dict[images_placeholders[i]] = gt_visuals[i]

    # reverse map vis order
    for i in range(len(assign["ds_factors"])):
        feed_dict[images_placeholders[i]] = np.concatenate([gt_visuals[i], map_visuals[i]])




    for key in feed_dict.keys():
        feed_dict[key] = np.expand_dims(feed_dict[key], 0)

    if blob["helper"] is not None:
        feed_dict[images_placeholders[len(images_placeholders)-2]] = (blob["helper"]/np.max(blob["helper"])*255).astype(np.uint8)
    else:
        feed_dict[images_placeholders[len(images_placeholders) - 2]] = np.zeros(blob["data"].shape,dtype=np.uint8)

    if blob["data"].shape[3] == 1:
        img_data = np.concatenate([blob["data"],blob["data"],blob["data"]],-1).astype(np.uint8)
    else:
        img_data = blob["data"].astype(np.uint8)
    feed_dict[images_placeholders[len(images_placeholders)-1]] = img_data



    return feed_dict


def get_gt_placeholders(assign, imdb):
    gt_dim = assign["stamp_func"][1](None, assign["stamp_args"], nr_classes)
    return [tf.placeholder(tf.float32, shape=[None, None, None, gt_dim]) for x in assign["ds_factors"]]



def get_config_id(assign):
    return assign["stamp_func"][0]+"_"+ assign["stamp_args"]["loss"]



def get_checkpoint_dir(args):
    # assemble path
    if "DeepScores" in args.dataset:
        image_mode = "music"
    elif "MUSICMA" in args.dataset:
        image_mode = "music_handwritten"
    else:
        image_mode = "realistic"
    tbdir = cfg.EXP_DIR + "/" + image_mode +"/"+"pretrain_lvl_"+args.pretrain_lvl+"/" + args.model
    if not os.path.exists(tbdir):
        os.makedirs(tbdir)
    runs_dir = os.listdir(tbdir)
    if args.continue_training == "True":
        tbdir = tbdir + "/" + "run_" + str(len(runs_dir)-1)
    else:
        tbdir = tbdir+"/"+"run_"+str(len(runs_dir))
        os.makedirs(tbdir)
    return tbdir


def get_training_roidb(imdb, use_flipped):
  """Returns a roidb (Region of Interest database) for use in training."""
  if use_flipped:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def save_objectness_function_handles(args, imdb):
    FUNCTION_MAP = {'stamp_directions':stamp_directions,
                    'stamp_energy': stamp_energy,
                    'stamp_class': stamp_class,
                    'stamp_bbox': stamp_bbox
                    }

    for obj_setting in args.training_assignements:
        obj_setting["stamp_func"] = [obj_setting["stamp_func"], FUNCTION_MAP[obj_setting["stamp_func"]]]

    return args

def load_database(args):
    print("Setting up image database: " + args.dataset)
    imdb = get_imdb(args.dataset)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb, args.use_flipped == "True")
    print('{:d} roidb entries'.format(len(roidb)))

    if args.dataset_validation != "no":
        print("Setting up validation image database: " + args.dataset_validation)
        imdb_val = get_imdb(args.dataset_validation)
        print('Loaded dataset `{:s}` for validation'.format(imdb_val.name))
        roidb_val = get_training_roidb(imdb_val, False)
        print('{:d} roidb entries'.format(len(roidb_val)))
    else:
        imdb_val = None
        roidb_val = None


    data_layer = RoIDataLayer(roidb, imdb.num_classes)

    if roidb_val is not None:
        data_layer_val = RoIDataLayer(roidb_val, imdb_val.num_classes, random=True)

    return imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val

def get_nr_classes():
    return nr_classes
