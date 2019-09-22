import tensorflow as tf
import numpy as np

def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
        focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: logits is probability before softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal Loss for Dense Object Detection, 130(4), 485–491.
        https://doi.org/10.1016/j.ajodo.2005.02.022
        :param labels: ground truth labels, shape of [batch_size]
        :param logits: model's output, shape of [batch_size, num_cls]
        :param gamma:
        :param alpha:
        :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.convert_to_tensor(labels, tf.float32)
    logits = tf.convert_to_tensor(logits, tf.float32)

    logits = tf.nn.softmax(logits, dim=-1)
    model_out = tf.add(logits, epsilon)

    ce = tf.multiply(labels, -tf.log(model_out))
    weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return reduced_fl

def get_loss(logits, tgt, kwargs={}):
    """
    Constructs the loss function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    n_class = tf.shape(logits)[3]
    loss_name = kwargs.get("loss_name", "cross_entropy")
    act_name = kwargs.get("act_name", "softmax")

    if act_name is "softmax":
        act = tf.nn.softmax
    if act_name is "sigmoid":
        act = tf.nn.sigmoid
    if act_name is "identity":
        act = tf.identity

    if not loss_name is "cross_entropy":
        prediction = act(logits)

    print("Loss Type: " + loss_name)

    if loss_name is "cross_entropy":
        class_weights = kwargs.get("class_weights", None)
        if class_weights is not None:
            print("Class Weights: " + str(class_weights))
            class_weights_tf = tf.constant(np.array(class_weights, dtype=np.float32))
            flat_logits = tf.reshape(logits, [-1, n_class])
            flat_labels = tf.reshape(tgt, [-1, n_class])
            weight_map = tf.multiply(flat_labels, class_weights_tf)
            weight_map = tf.reduce_sum(weight_map, axis=1)

            if act_name is "softmax":
                loss_map = focal_loss(logits=flat_logits, labels=flat_labels)
            elif act_name is "sigmoid":
                loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

            weighted_loss = tf.multiply(loss_map, weight_map)
            loss = tf.reduce_mean(weighted_loss)

        else:
            flat_logits = tf.reshape(logits, [-1, n_class])
            flat_labels = tf.reshape(tgt, [-1, n_class])

            if act_name is "softmax":
                loss_map = focal_loss(logits=flat_logits, labels=flat_labels)
            elif act_name is "sigmoid":
                loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

            
            loss = tf.reduce_mean(loss_map)
            final_loss = None

    elif loss_name is "dice":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        eps = 1e-5
        if ignore_last_channel:
            prediction = tf.slice(prediction, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            tgt = tf.slice(tgt, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
        intersection = tf.reduce_sum(prediction * tgt)
        union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(tgt)
        loss = -(2 * intersection / (union))

    elif loss_name is "dice_mean":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        eps = 1e-5
        uPred = tf.unstack(prediction,axis=3)
        uTgt = tf.unstack(tgt,axis=3)
        if ignore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        dices=[]
        for aCH in range(0,len(uPred)):
            intersection = tf.reduce_sum(uPred[aCH] * uTgt[aCH])
            union = eps + tf.reduce_sum(uPred[aCH]) + tf.reduce_sum(uTgt[aCH])
            aDice = -(2 * intersection / (union))
            dices.append(aDice)
        loss = tf.reduce_mean(dices)

    elif loss_name is "mse":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if ignore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        for aCH in range(0, len(uPred)):
            aRMSE = tf.losses.mean_squared_error(uTgt[aCH], uPred[aCH])
            mses.append(aRMSE)
        loss = 0.0

        class_weights = kwargs.get("class_weights", None)
        if class_weights is not None:
            print("Class Weights: " + str(class_weights))
            norm = 0
            for aCH in range(0,len(mses)):
                loss += class_weights[aCH]*mses[aCH]
                norm += class_weights[aCH]
        else:
            for aCH in range(0,len(mses)):
                loss += mses[aCH]

    elif loss_name is "mse_mean":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if ignore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        sums=[]
        globSum = 0.0
        for aCH in range(0, len(uPred)):
            aRMSE = tf.losses.mean_squared_error(uTgt[aCH],uPred[aCH])
            mses.append(aRMSE)
            aSum = tf.reduce_sum(uTgt[aCH])
            globSum += aSum
            sums.append(aSum)
        loss = tf.cond(tf.equal(globSum, 0.0),
                              lambda: tf.reduce_mean(mses),
                              lambda: get_weighted_mean(mses, sums, globSum))
    elif loss_name is "nse":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        eps = 1e-5
        norm = eps + tf.reduce_sum(tgt)+tf.reduce_sum(prediction)
        if ignore_last_channel:
            prediction = tf.slice(prediction, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            tgt = tf.slice(tgt, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            norm = eps + tf.reduce_sum(tgt) + tf.reduce_sum(prediction)
        loss = tf.reduce_sum(tf.squared_difference(tgt, prediction))
        loss = loss/norm

    elif loss_name is "nse_mean":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)
        eps = 1e-5
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if ignore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        for aCH in range(0, len(uPred)):
            norm = eps + tf.reduce_sum(uTgt[aCH]) + tf.reduce_sum(uPred[aCH])
            aRMSE = tf.reduce_sum(tf.squared_difference(uTgt[aCH], uPred[aCH]))/norm
            mses.append(aRMSE)
        loss = tf.reduce_mean(mses)

    elif loss_name is "combined":
        ignore_last_channel = kwargs.get("ignore_last_channel", True)

        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(tgt, [-1, n_class])
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                      labels=flat_labels))
        eps = 1e-5
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if ignore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        dices = []
        for aCH in range(0, len(uPred)):
            intersection = tf.reduce_sum(uPred[aCH] * uTgt[aCH])
            union = eps + tf.reduce_sum(uPred[aCH]) + tf.reduce_sum(uTgt[aCH])
            aDice = -(2 * intersection / (union))
            dices.append(aDice)
        loss2 = tf.reduce_mean(dices)
        loss = loss1 + loss2

    elif loss_name is "cross_entropy_sum":
        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(tgt, [-1, n_class])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                      labels=flat_labels))
    else:
        raise ValueError("Unknown loss function: " % loss_name)

    regularizer = kwargs.get("regularizer", None)
    if regularizer is not None:
        losses = []
        for var in tf.trainable_variables():
            if 'weights' in var.op.name:
                losses.append(tf.nn.l2_loss(var))
        loss += (regularizer * tf.add_n(losses))

    return loss, final_loss

def get_weighted_mean(mses, sums, globSum):
    loss = 0.0
    for aCH in range(0, len(sums)):
        aLoss = (1.0-sums[aCH]/globSum) * mses[aCH]
        loss += aLoss
    return loss

def get_mean_iou(pred_class, tgt_class, num_class, ignore_class_id=-1):
    pred = tf.reshape(pred_class, [-1, ])
    gt = tf.reshape(tgt_class, [-1, ])

    if ignore_class_id >= 0:
        weights = tf.cast(tf.not_equal(gt, ignore_class_id), tf.int32)
    else:
        weights = tf.ones_like(gt)

    # mIoU
    # mIoU, update_op = tf.metrics.mean_iou(pred, gt, num_classes=num_class, weights=weights, name='m_metrics')
    # mIoU = tf.reduce_mean(tf.cast(pred == gt, dtype='float32'))
    mIoU, update_op = tf.metrics.accuracy(pred, gt, weights=weights, name='m_metrics')

    return mIoU, update_op