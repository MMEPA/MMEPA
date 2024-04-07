import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import logging

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    # np.round(数据, decimal=保留的小数位数). 原则：对于浮点型数据，四舍六入，正好一半取偶数。


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    # .view()表示重组维度。参数为-1表示维度推断。
    # .cpu()将数据的处理设备从其他设备, 如.cuda()拿到cpu上, 不会改变变量类型, 转换后仍然是Tensor变量。
    # .detach()返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不同之处只是requirse_grad为false.
    # 得到的这个tensir永远不需要计算器梯度，不具有grad.
    # 使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变
    # .numpy()将Tensor变量转换为ndarray变量.
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 非0值的test_truth索引

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)  # 进行裁剪, 让值限定在-3~ +3
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]  # 皮尔逊乘积矩阵
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth_non0 = test_truth[non_zeros] > 0
    binary_preds_non0 = test_preds[non_zeros] > 0
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')
    
    
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score all/non0: {}/{} over {}/{}".format(np.round(f_score,4), np.round(f_score_non0,4), binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
    print("Accuracy all/non0: {}/{}".format(np.round(acc_2,4), np.round(acc_2_non0,4)))

    print("-" * 50)
    
    logging.info(f"MAE: {mae} ")
    logging.info(f"Correlation Coefficient: {corr}")
    logging.info(f"mult_acc_7: {mult_a7}")
    logging.info(f"mult_acc_5: {mult_a5}")
    logging.info("F1 score all/non0: {}/{} over {}/{}".format(np.round(f_score,4), np.round(f_score_non0,4), binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
    logging.info("Accuracy all/non0: {}/{}".format(np.round(acc_2,4), np.round(acc_2_non0,4)))

    return {'mae':mae, 'corr':corr, 'mult':mult_a7, 'f1':f_score_non0, 'acc2':acc_2_non0}

def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)
