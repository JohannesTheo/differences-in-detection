from tidecv import TIDE, datasets


def get_tide_results(ann_file, results_file, iou_type, threshold=0.5):

    gt = datasets.COCO(ann_file)
    preds = datasets.COCOResult(results_file, iou_type)

    tide = TIDE()
    run = tide.evaluate(gt=gt, preds=preds, mode='bbox', pos_threshold=0.5, name=str(0.5))

    metrics = {}
    metrics['AP'] = run.ap
    metrics['main'] = {}
    metrics['special'] = {}

    main_errors = run.fix_main_errors()
    special_errors = run.fix_special_errors()

    for error, value in main_errors.items():
        metrics['main'][error.short_name] = value

    for error, value in special_errors.items():
        metrics['special'][error.short_name] = value

    return metrics
