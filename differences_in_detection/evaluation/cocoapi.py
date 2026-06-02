import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _coco_summarize(params, eval, ap=1, iouThr=None, areaRng='all', maxDets=100, verbose=False):
    """
    Modified version of _summarize, instead of self.params and self.eval we pass these as arguments
    See: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L427

    :param params: cocoEval.params
    :param eval: cocoEval.eval
    :param verbose: only print if requested
    :return: the requested statistic and per_class statistic as dict {'mean':s, 'per_class': cs_list }
    """

    p = params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])

        # NEW: add per class stats
        mean_cs = []
        for idx in range(80):
            cs = s[:,:,idx,:] if ap == 1 else s[:,idx,:]
            mean_cs.append(np.mean(cs[cs > -1]))

    if verbose:
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s, mean_cs


def coco_summary(coco_eval, keep=['AP', 'AP50', 'AR100'], verbose=False):

    summary_dict = {}
    params, eval = coco_eval.params, coco_eval.eval

    # Add default COCO metrics (as in the standard 'summary')
    short_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100','ARs', 'ARm', 'ARl',
                   'AR100_50', 'AR100_75']
    stats = np.zeros((14,))
    stats_cs = np.zeros((14,80))
    stats[0], stats_cs[0] = _coco_summarize(params, eval, 1, verbose=verbose)
    stats[1], stats_cs[1] = _coco_summarize(params, eval, 1, iouThr=.5, maxDets=params.maxDets[2], verbose=verbose)
    stats[2], stats_cs[2] = _coco_summarize(params, eval, 1, iouThr=.75, maxDets=params.maxDets[2], verbose=verbose)
    stats[3], stats_cs[3] = _coco_summarize(params, eval, 1, areaRng='small', maxDets=params.maxDets[2], verbose=verbose)
    stats[4], stats_cs[4] = _coco_summarize(params, eval, 1, areaRng='medium', maxDets=params.maxDets[2], verbose=verbose)
    stats[5], stats_cs[5] = _coco_summarize(params, eval, 1, areaRng='large', maxDets=params.maxDets[2], verbose=verbose)
    stats[6], stats_cs[6] = _coco_summarize(params, eval, 0, maxDets=params.maxDets[0], verbose=verbose)
    stats[7], stats_cs[7] = _coco_summarize(params, eval, 0, maxDets=params.maxDets[1], verbose=verbose)
    stats[8], stats_cs[8] = _coco_summarize(params, eval, 0, maxDets=params.maxDets[2], verbose=verbose)
    stats[9], stats_cs[9] = _coco_summarize(params, eval, 0, areaRng='small', maxDets=params.maxDets[2], verbose=verbose)
    stats[10], stats_cs[10] = _coco_summarize(params, eval, 0, areaRng='medium', maxDets=params.maxDets[2], verbose=verbose)
    stats[11], stats_cs[11] = _coco_summarize(params, eval, 0, areaRng='large', maxDets=params.maxDets[2], verbose=verbose)
    # NEW: not part of standard coco summary
    stats[12], stats_cs[12] = _coco_summarize(params, eval, 0, iouThr=.5,  maxDets=params.maxDets[2], verbose=verbose)
    stats[13], stats_cs[13] = _coco_summarize(params, eval, 0, iouThr=.75, maxDets=params.maxDets[2], verbose=verbose)

    # Only keep the requested ones
    summary_dict.update(dict([(name, {'mean': stat, 'per_class': stat_cs})
                              for name, stat, stat_cs in
                              zip(short_names, stats, stats_cs) if name in keep]))

    return summary_dict


def get_cocoapi_results(ann_file, results_file, iou_type, print_standard_summary=True):

    assert iou_type in ['bbox', 'segm']

    with open(results_file,'r') as f:
        predictions = json.load(f)

        # TODO: add explanation
        if iou_type == 'segm':
            for x in predictions:
                x.pop('bbox')

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(predictions)
    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.evaluate()
    cocoEval.accumulate()

    if print_standard_summary:
        cocoEval.summarize()

    return coco_summary(cocoEval)
