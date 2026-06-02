import json
import copy
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from tidecv import functions as f
from tidecv import TIDE, TIDERun, TIDEExample
from tidecv.data import Data
from tidecv.ap import ClassedAPDataObject, APDataObject
from tidecv.errors.qualifiers import Qualifier
from tidecv.errors.error import BestGTMatch
from tidecv.errors.main_errors import (
    ClassError, BoxError, OtherError, DuplicateError,
    BackgroundError, MissedError, FalsePositiveError,
    FalseNegativeError)


# =========================================================
# Extending TIDE functionality for DND
# =========================================================

class Data_DND(Data):

    def __init__(self, name, max_dets = 100):
        super().__init__(name, max_dets)

    def _add(self, ann_id:int, image_id:int, class_id:int, box:object=None, mask:object=None,  boundary:object=None, score:float=1, ignore:bool=False):
        """ Add a data object to this collection. You should use one of the below functions instead. """
        self._make_default_class(class_id)
        self._make_default_image(image_id)
        new_id = len(self.annotations)
        #print(ann_id, new_id)

        self.annotations.append({
            '_id'   : new_id,
            'ann_id': ann_id,  # NOTE: NEW - only because of this lol, TODO: could do this another way maybe
            'score' : score,
            'image' : image_id,
            'class' : class_id,
            'bbox'  : self._prepare_box(box),
            'mask'  : self._prepare_mask(mask),
            #'boundary': self._prepare_boundary(boundary),
            'ignore': ignore,
        })

        self.images[image_id]['anns'].append(new_id)

    def add_ground_truth(self, ann_id:int, image_id:int, class_id:int, box:object=None, mask:object=None, boundary:object=None):
        # NOTE: NEW: ann_id
        """ Add a ground truth. If box or mask is None, this GT will be ignored for that mode. """
        self._add(ann_id, image_id, class_id, box, mask, boundary)

    def add_ignore_region(self, ann_id:int, image_id:int, class_id:int=None, box:object=None, mask:object=None, boundary:object=None):
        """
        Add a region inside of which background detections should be ignored.
        You can use these to mark a region that has deliberately been left unannotated
        (e.g., if is a huge crowd of people and you don't want to annotate every single person in the crowd).

        If class_id is -1, this region will match any class. If the box / mask is None, the region will be the entire image.
        """
        # NOTE: NEW: ann_id
        self._add(ann_id, image_id, class_id, box, mask, boundary, ignore=True)


def TIDE_COCO_DND(annotations: str | dict, name:str=None) -> Data:
    """
    Loads ground truth from a COCO-style annotation file or dict.

    NOTE: copy of tide COCO: accepts already loaded annotation dicts (in addition to file name)
          also checks if ann has segmentation to allow datasets without masks, e.g. coco-c, nao, ...
    """
    if isinstance(annotations, str):  # i.e. annotation_file
        with open(annotations, 'r') as json_file:
            cocojson = json.load(json_file)
    else:
        cocojson = annotations
    if name is None: name = cocojson['info']['description']

    images = cocojson['images']
    anns   = cocojson['annotations']
    cats   = cocojson['categories'] if 'categories' in cocojson else None

    # Add everything from the coco json into our data structure
    data = Data_DND(name, max_dets=100)

    image_lookup = {}

    for idx, image in enumerate(images):
        image_lookup[image['id']] = image
        data.add_image(image['id'], image['file_name'])

    if cats is not None:
        for cat in cats:
            data.add_class(cat['id'], cat['name'])

    for ann in anns:
        ann_id = ann['id']  # NOTE: NEW - only because of this lol, TODO: could do this another way maybe
        image  = ann['image_id']
        _class = ann['category_id']
        box    = ann['bbox']
        if ann['segmentation']:
            mask = f.toRLE(ann['segmentation'], image_lookup[image]['width'], image_lookup[image]['height'])
        else:
            mask = None
        boundary = ann['boundary'] if 'boundary' in ann.keys() else None  # we assume boundaries are RLEs when given

        if ann.get('iscrowd', False):  # COCO crowds are ignored (somme ann files have 'iscrowd' not set)
            data.add_ignore_region(ann_id, image, _class, box, mask, boundary)
        else:
            data.add_ground_truth(ann_id, image, _class, box, mask, boundary)

    return data


class BestGTMatch_DND(BestGTMatch):

    def __init__(self, pred, gt):
        self.pred = pred
        self.gt = gt

        if self.gt['used']:
            self.suppress = True
        else:
            self.suppress = False
            self.gt['usable_other_dnd'] = True

            score = self.pred['score']

            if not 'best_score_other_dnd' in self.gt:
                self.gt['best_score_other_dnd'] = -1

            if self.gt['best_score_other_dnd'] < score:
                self.gt['best_score_other_dnd'] = score
                self.gt['best_id_other_dnd'] = self.pred['_id']

    def fix(self):
        if self.suppress or self.gt['best_id_other_dnd'] != self.pred['_id'] or 'best_id' in self.gt:
            return None
        else:
            # NOTE: The default behavior of BestGTMatch would FIX the error by
            #       matching a GT as if class and localization were correct.
            #       This will change TIDE results for 'OtherError', but we only
            #       want to count them...
            #return (self.pred['score'], True, self.pred['info'])
            #
            #       ... so instead, we treat/FIX the 'matched' case like a 'MissedError',
            #       and use -2 as identifier, so we can distinguis them in TIDERun_DND!
            return -2


class OtherError_DND(OtherError):

    description = "This detection didn't fall into any of the other error categories. - DND version"
    short_name  = "Both"

    def __init__(self, pred:dict, gt:dict, ex):
        self.pred = pred
        self.gt = gt

        assert self.pred['class'] != self.gt['class'], (f"This should be a BoxError error: \n"
                                                        f"pred: {self.pred} \n"
                                                        f"gt:   {self.gt}")

        self.match = BestGTMatch_DND(pred, gt) if not self.gt['used'] else None

    def fix(self):
        if self.match is None:
            return None
        return self.pred['class'], self.match.fix()


class APDataObject_DND(APDataObject):
    def __init__(self):
        super().__init__()


    def get_ar(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score (NOTE: actually not relevant for recall)
        data_points = list(self.data_points.values())
        #data_points.sort(key=lambda x: -x[0])

        num_true  = len([datum for datum in data_points if datum[1]])
        recall    = num_true / self.num_gt_positives

        return recall*100


class ClassedAPDataObject_DND(ClassedAPDataObject):

    def __init__(self):
        self.objs = defaultdict(lambda: APDataObject_DND())

    def apply_qualifier(self, pred_dict:dict, gt_dict:dict) -> object:
        ret = ClassedAPDataObject_DND()

        for _class, obj in self.objs.items():
            pred_list = pred_dict[_class] if _class in pred_dict else set()
            gt_list   =   gt_dict[_class] if _class in   gt_dict else set()

            ret.objs[_class] = obj.apply_qualifier(pred_list, gt_list)

        return ret

    def get_mAP_cs(self) -> float:
        #aps = [x.get_ap() for x in self.objs.values() if not x.is_empty()]
        aps = [self.objs[idx].get_ap() for idx in sorted(self.objs.keys()) if not self.objs[idx].is_empty()]
        return aps

    def get_mAR(self) -> float:
        ars = [x.get_ar() for x in self.objs.values() if not x.is_empty()]
        if len(ars) == 0:
            return -1.0
        return sum(ars) / len(ars)

    def get_mAR_cs(self) -> float:
        #ars = [x.get_ar() for x in self.objs.values() if not x.is_empty()]
        ars = [self.objs[idx].get_ar() for idx in sorted(self.objs.keys()) if not self.objs[idx].is_empty()]
        return ars


class TIDE_DND(TIDE):

    _error_types = [ClassError, BoxError, OtherError_DND, DuplicateError, BackgroundError, MissedError]

    def __init__(self, pos_threshold = 0.5, background_threshold = 0.1, mode = TIDE.BOX):
        super().__init__(pos_threshold, background_threshold, mode)

    def evaluate(self, gt:Data, preds:Data, pos_threshold:float=None, background_threshold:float=None, 
                 mode:str=None, name:str=None, use_for_errors:bool=True, verbose:bool=True) -> TIDERun:
        pos_thresh = self.pos_thresh if pos_threshold        is None else pos_threshold
        bg_thresh  = self.bg_thresh  if background_threshold is None else background_threshold
        mode       = self.mode       if mode                 is None else mode
        name       = preds.name      if name                 is None else name

        run = TIDERun_DND(gt, preds, pos_thresh, bg_thresh, mode, gt.max_dets, use_for_errors, verbose=verbose)

        if use_for_errors:
            self.runs[name] = run

        return run

    def evaluate_range(self, gt:Data, preds:Data, thresholds:list=TIDE.COCO_THRESHOLDS, pos_threshold:float=None,
                            background_threshold:float=None, mode:str=None, name:str=None) -> dict:
        raise NotImplementedError()


class TIDERun_DND(TIDERun):

    _temp_vars = ['best_score', 'best_id', 'used', 'matched_with', '_idx', 'usable',
                  'usable_other_dnd', 'best_score_other_dnd', 'best_id_other_dnd' ]

    def __init__(self, gt, preds, pos_thresh, bg_thresh, mode, max_dets, run_errors = True, verbose = True, dnd=True):
        self.gt     = gt
        self.preds  = preds

        self.errors     = []
        self.error_dict = {_type: [] for _type in TIDE_DND._error_types}
        self.ap_data = ClassedAPDataObject_DND()
        self.qualifiers = {}

        # A list of false negatives per class
        self.false_negatives = {_id: [] for _id in self.gt.classes}

        self.pos_thresh = pos_thresh
        self.bg_thresh  = bg_thresh
        self.mode       = mode
        self.max_dets   = max_dets
        self.run_errors = run_errors
        self.verbose = verbose

        self._run()


    def _eval_image(self, preds:list, gt:list):

        for truth in gt:
            if not truth['ignore']:
                self.ap_data.add_gt_positives(truth['class'], 1)

        if len(preds) == 0:
            # There are no predictions for this image so add all gt as missed
            for truth in gt:
                if not truth['ignore']:
                    self.ap_data.push_false_negative(truth['class'], truth['_id'])

                    if self.run_errors:
                        self._add_error(MissedError(truth))
                        self.false_negatives[truth['class']].append(truth)
            return

        ex = TIDEExample(preds, gt, self.pos_thresh, self.mode, self.max_dets, self.run_errors)
        preds = ex.preds # In case the number of predictions was restricted to the max

        for pred_idx, pred in enumerate(preds):

            pred['info'] = {'iou': pred['iou'], 'used': pred['used']}
            if pred['used']: pred['info']['matched_with'] = pred['matched_with']

            if pred['used'] is not None:
                self.ap_data.push(pred['class'], pred['_id'], pred['score'], pred['used'], pred['info'])

            # ----- ERROR DETECTION ------ #
            # This prediction is a negative (or ignored), let's find out why
            if self.run_errors and (pred['used'] == False or pred['used'] == None):
                # Test for BackgroundError
                if len(ex.gt) == 0: # Note this is ex.gt because it doesn't include ignore annotations
                    # There is no ground truth for this image, so just mark everything as BackgroundError
                    self._add_error(BackgroundError(pred))
                    continue

                # Test for BoxError
                idx = ex.gt_cls_iou[pred_idx, :].argmax()
                if self.bg_thresh <= ex.gt_cls_iou[pred_idx, idx] <= self.pos_thresh:
                    # This detection would have been positive if it had higher IoU with this GT
                    self._add_error(BoxError(pred, ex.gt[idx], ex))
                    continue

                # Test for ClassError
                idx = ex.gt_noncls_iou[pred_idx, :].argmax()
                if ex.gt_noncls_iou[pred_idx, idx] >= self.pos_thresh:
                    # This detection would have been a positive if it was the correct class
                    self._add_error(ClassError(pred, ex.gt[idx], ex))
                    continue

                # Test for DuplicateError
                idx = ex.gt_used_cls[pred_idx, :].argmax()
                if ex.gt_used_cls[pred_idx, idx] >= self.pos_thresh:
                    # The detection would have been marked positive but the GT was already in use
                    suppressor = self.preds.annotations[ex.gt[idx]['matched_with']]
                    self._add_error(DuplicateError(pred, suppressor))
                    continue

                # Test for BackgroundError
                idx = ex.gt_iou[pred_idx, :].argmax()
                if ex.gt_iou[pred_idx, idx] <= self.bg_thresh:
                    # This should have been marked as background
                    self._add_error(BackgroundError(pred))
                    continue

                # NOTE: NEW, instead of the default OtherError, we also
                #       match the prediction to the closest unused gt.
                #
                #       We do NOT change how the error is fixed though!
                #
                #       See OtherError_DND for details...
                c_idx = ex.gt_noncls_iou[pred_idx, :].argmax()
                g_idx = ex.gt_iou[pred_idx, :].argmax()
                assert c_idx == g_idx
                c_iou = ex.gt_noncls_iou[pred_idx, idx]
                g_iou = ex.gt_iou[pred_idx, idx]
                assert c_iou == g_iou
                assert self.bg_thresh < g_iou <= self.pos_thresh
                self._add_error(OtherError_DND(pred, ex.gt[g_idx], ex))
                #self._add_error(OtherError(pred))

        for truth in gt:
            # If the GT wasn't used in matching, meaning it's some kind of false negative
            if not truth['ignore'] and not truth['used']:
                self.ap_data.push_false_negative(truth['class'], truth['_id'])

                if self.run_errors:
                    self.false_negatives[truth['class']].append(truth)

                    # The GT was completely missed, no error can correct it
                    # Note: 'usable' is set in error.py
                    if not truth['usable']:
                        self._add_error(MissedError(truth))


    def fix_errors(self, condition=lambda x: False, transform=None, false_neg_dict:dict=None,
                   ap_data:ClassedAPDataObject_DND=None,disable_errors:bool=False) -> ClassedAPDataObject_DND:
        """ Returns a ClassedAPDataObject_DND where all errors given the condition returns True are fixed. """
        if ap_data is None:
            ap_data = self.ap_data

        gt_pos = ap_data.get_gt_positives()
        new_ap_data = ClassedAPDataObject_DND()

        # NOTE: NEW, collect gt ann_ids of fixed and unfixed dts
        fixed_errors_gt_dt = {}
        _unused_errors_gt_dt = []

        # Potentially fix every error case
        for error in self.errors:
            if error.disabled:
                continue

            _id = error.get_id()
            _cls, data_point = error.original   # NOTE: by default everything is unfixed

            if condition(error):
                _cls, data_point = error.fixed  # NOTE: but for a tested error, us fixed

                if disable_errors:
                    error.disabled = True

                # NOTE: NEW, collect gt ids for fixed ClassError and BoxError
                #       Only these two return not None since they match an un-
                #       match gt! MissedError is collected and fixed below!
                if data_point is not None:
                    if hasattr(error, 'match') and not isinstance(error, OtherError_DND):
                        # not sure if necessary but let's do these checks
                        assert hasattr(error, 'gt')
                        assert hasattr(error.match, 'gt')
                        assert error.gt['ann_id'] == error.match.gt['ann_id']
                        assert not data_point[-1]['used']  # (info dict)
                        gt_id = error.gt['ann_id']
                        assert gt_id not in fixed_errors_gt_dt.keys()
                        fixed_errors_gt_dt[gt_id] = copy.deepcopy(error.pred)
                    else:
                        # Only Missed Error and OtherError_DND, these
                        # will be counted and fixed below...
                        assert isinstance(data_point, int)
                else:
                    # In this case, the candidate is either not used to fix an error OR it is one of
                    # 'OtherError', 'DuplicateError' or 'BackgroundError'. These are fixed as None
                    # by default, i.e. simply supressed and not added to new_ap_data below...
                    if hasattr(error, 'gt'):
                        # TODO: should we distinguish between these cases?
                        # if error.match is None:
                        #     assert error.gt['used']  # gt required no fix
                        # else:
                        #     assert not error.gt['used']
                        _unused_errors_gt_dt.append((error.gt['ann_id'], copy.deepcopy(error.pred)))
                    else:
                        assert not hasattr(error, 'match')
                        assert any([
                            isinstance(error, DuplicateError),
                            isinstance(error, OtherError),
                            isinstance(error, BackgroundError),
                        ])

                # Specific for MissingError (or anything else that affects #GT)
                if isinstance(data_point, int):

                    # NOTE: NEW, only do this for MissingError (OtherError_DND is -2)
                    if data_point == -1:
                        gt_pos[_cls] += data_point

                    data_point = None
                    #fixed_errors_gt_dt.append((error.gt['ann_id'], copy.deepcopy(error.pred)))
                    gt_id = error.gt['ann_id']
                    assert gt_id not in fixed_errors_gt_dt.keys()
                    fixed_errors_gt_dt[gt_id] = None # copy.deepcopy(error.pred)

            # NOTE: if data_point is None, the pred/error will be ignored in the new AP calculation!
            if data_point is not None:
                if transform is not None:
                    data_point = transform(*data_point)
                new_ap_data.push(_cls, _id, *data_point)

        # Add back all the correct ones
        for k in gt_pos.keys():
            for _id, (score, correct, info) in ap_data.objs[k].data_points.items():
                if correct:
                    if transform is not None:
                        score, correct, info = transform(score, correct, info)
                    new_ap_data.push(k, _id, score, correct, info)

        # Add the correct amount of GT positives, and also subtract if necessary
        for k, v in gt_pos.items():
            # In case you want to fix all false negatives without affecting precision
            if false_neg_dict is not None and k in false_neg_dict:
                v -= len(false_neg_dict[k])
            new_ap_data.add_gt_positives(k, v)

        return new_ap_data, fixed_errors_gt_dt, _unused_errors_gt_dt


    def fix_main_errors(self, progressive:bool=False, error_types:list=None, qual:Qualifier=None) -> dict:
        ap_data = self.ap_data
        last_ap = self.ap

        if qual is None:
            qual = Qualifier('', None)

        if error_types is None:
            error_types = TIDE._error_types

        errors = {}
        fixed_gt = {}
        fixed_gt_dt_maps = {}
        _unused_err_gt = {}

        for error in error_types:
            _ap_data, _fixed_errors_gt_dt, _unused_errors_gt_dt = self.fix_errors(
                qual._make_error_func(error), ap_data=ap_data, disable_errors=progressive)

            new_ap = _ap_data.get_mAP()
            # If an error is negative that means it's likely due to binning differences, so just
            # Ignore the negative by setting it to 0.
            errors[error] = max(new_ap - last_ap, 0)
            fixed_gt[error.short_name] = list(_fixed_errors_gt_dt.keys())
            fixed_gt_dt_maps[error.short_name] = _fixed_errors_gt_dt
            _unused_err_gt[error.short_name] = _unused_errors_gt_dt

            if progressive:
                last_ap = new_ap
                ap_data = _ap_data

        if progressive:
            for error in self.errors:
                error.disabled = False

        return errors, fixed_gt, fixed_gt_dt_maps #, _unused_err_gt

    def fix_special_errors(self, qual=None) -> dict:
        FPE, _, _ = self.fix_errors(transform=FalsePositiveError.fix)
        FNE, _, _ = self.fix_errors(false_neg_dict=self.false_negatives)
        return {
            FalsePositiveError: FPE.get_mAP() - self.ap,
            FalseNegativeError: FNE.get_mAP() - self.ap
            }


# =========================================================
# Calculation of DND Subsets
# =========================================================


from tidecv import datasets


def get_dnd_matches(tide_dnd_run, verbose=False):

    run = tide_dnd_run

    gt_anns = run.gt.annotations
    dt_anns = run.preds.annotations
    all_gt = [gt['ann_id'] for gt in gt_anns if not gt['ignore']]
    matched_gt = []
    gt_dt_map = {}
    for dt in dt_anns:
        if 'matched_with' in dt and dt['used']:
            gt_id = gt_anns[dt['matched_with']]['ann_id']
            matched_gt.append(gt_id)
            gt_dt_map[gt_id] = (copy.deepcopy(dt), 'Matched')
    assert len(matched_gt) == len(gt_dt_map.keys())

    #matched_gt_dt = dict([(gt_anns[dt['matched_with']]['ann_id'], dt) for dt in dt_anns if 'matched_with' in dt and dt['used']])

    _metrics, fixed_gt, fixed_gt_dt_maps = run.fix_main_errors()
    cls = fixed_gt['Cls']
    loc = fixed_gt['Loc']
    both = fixed_gt['Both']  # Both is a subset of Miss error
    miss = fixed_gt['Miss']
    fixed_errors = [*cls, *loc, *miss]
    assert len(both) <= len(miss)

    # add fixed dt gt pairs to gt_dt_map
    for error_type in ['Cls', 'Loc', 'Both', 'Miss']:
        e_gt_dt_map = fixed_gt_dt_maps[error_type]
        for gt, dt in e_gt_dt_map.items():
            assert gt not in matched_gt
            assert gt in fixed_gt[error_type]

            # this can only happen for Miss errors because 
            # Both is a subset of Miss and added first!
            if gt in gt_dt_map.keys():
                assert error_type == 'Miss', gt_dt_map[gt]
                assert dt is None
                continue

            gt_dt_map[gt] = (dt, error_type)

    _dt_ids = [dt[0]['_id'] for gt, dt in gt_dt_map.items() if dt[0] is not None]
    assert len(_dt_ids) == len(set(_dt_ids))

    # No duplicate gt ids
    for s in [all_gt, matched_gt, cls, loc, both, miss, fixed_errors]:
        assert len(s) == len(set(s))

    # Distinct sets
    assert set(cls) & set(loc)  == set()
    assert set(cls) & set(both) == set()
    assert set(cls) & set(miss) == set()

    assert set(loc) & set(both) == set()
    assert set(loc) & set(miss) == set()

    assert set(matched_gt) & set(fixed_errors) == set()

    # Subsets
    assert set(both).issubset(set(miss))
    assert set(matched_gt).issubset(set(all_gt))
    assert set(fixed_errors).issubset(set(all_gt))

    # Does it sum?
    assert set(all_gt) - set(matched_gt) == set(fixed_errors)  # fixed errors must cover all unmatched gt
    assert set(all_gt) == set(matched_gt) | set(fixed_errors)

    # if verbose:
    #     print(f"{model} @ {threshold}")
    #     print(f"Number of GT anns    : {len(all_gt)} ({len(gt_anns)} with crowd))")
    #     print(f"Number of matched    : {len(matched_gt)}")
    #     print(f"Number of not-matched: {len(fixed_errors)}")
    #     print(f"                       {len(fixed_errors)+len(matched_gt)} ({len(fixed_errors)+len(matched_gt) == len(all_gt)})")
    #     print(f"Number of error types:")
    #     print(f"               - Cls : {len(cls)}")
    #     print(f"               - Loc : {len(loc)}")
    #     print(f"               - Miss: {len(miss)} ({len(both)} 'Both')")
    #     print(f"                       {len(cls) + len(loc) + len(miss)} ({len(cls) + len(loc) + len(miss)==len(fixed_errors)})")

    return all_gt, matched_gt, fixed_gt, gt_dt_map


def flatten_fixed_errros(err_dict):
    all_fixed_errros = []
    for en in ['Cls', 'Loc', 'Miss']:
        all_fixed_errros.extend(err_dict[en])
    assert len(all_fixed_errros) == len(set(all_fixed_errros))
    return set(all_fixed_errros)

def get_errror_distribution(ex, err_dict):
    ex_cat = []
    for fe_id in sorted(list(ex)):
        found = False
        for error_type in ['Cls', 'Loc', 'Both', 'Miss']:  # Check 'Both' before 'Miss' as it is a subset!
            if fe_id in err_dict[error_type]:
                found = True
                ex_cat.append(error_type)
                break
        if not found:
            raise ValueError(f"fixed GT ID: '{fe_id}' not in err_dict")
    return ex_cat


def dnd_compare(gt, m1, m2, e1, e2, d1, d2, verbose=False):
    # TODO: for every gt, map d1 and d2, including the status "Matched" or Error Type!

    # Definitions from matched GTs (as in paper):
    A = set(m1) & set(m2)
    B = set(m1) - A
    C = set(m2) - A
    D = set(gt) - (set(m1) | set(m2))

    EoB = D | C
    EoC = D | B
    ExB = EoB - D
    ExC = EoC - D
    assert EoB == set(gt) - set(m1) == set(gt) - (A | B)
    assert EoC == set(gt) - set(m2) == set(gt) - (A | C)
    assert ExB == C
    assert ExC == B

    # Check against IDs from fixed errors
    _EoB = flatten_fixed_errros(e1)
    _EoC = flatten_fixed_errros(e2)
    _ExB = _EoB - _EoC
    _ExC = _EoC - _EoB
    _D = _EoB  & _EoC
    assert EoB == _EoB
    assert EoC == _EoC
    assert ExB == _ExB
    assert ExC == _ExC
    assert _D == D

    # Error Distributions
    EoB_dist = get_errror_distribution(EoB, e1)
    EoC_dist = get_errror_distribution(EoC, e2)
    ExB_dist = get_errror_distribution(ExB, e1)
    ExC_dist = get_errror_distribution(ExC, e2)
    D1_dist = get_errror_distribution(D, e1)
    D2_dist = get_errror_distribution(D, e2)
    assert len(EoB) == sum(np.unique(EoB_dist, return_counts=True)[-1])
    assert len(EoC) == sum(np.unique(EoC_dist, return_counts=True)[-1])
    assert len(ExB) == sum(np.unique(ExB_dist, return_counts=True)[-1])
    assert len(ExC) == sum(np.unique(ExC_dist, return_counts=True)[-1])
    assert len(D1_dist) == len(D2_dist)
    assert len(D1_dist) + sum(np.unique(ExB_dist, return_counts=True)[-1]) == len(EoB)
    assert len(D2_dist) + sum(np.unique(ExC_dist, return_counts=True)[-1]) == len(EoC)

    gt_dt_map = {
        gt_id: (d1[gt_id], d2[gt_id]) for gt_id in gt
    }
    dnd_dict = {
        "GT": gt, "A": A, "B": B, "C": C, "D": D,
        "EoB": EoB, "EoC": EoC, "ExB": ExB, "ExC": ExC,
        "categorical_distributions": {
            "EoB": EoB_dist, "EoC": EoC_dist, "ExB": ExB_dist, "ExC": ExC_dist,
            "D1": D1_dist, "D2": D2_dist,
        },
        "gt_dt_map": gt_dt_map
    }

    if verbose:
        print("GT:", len(gt),
            "\nA:", len(A),
            "\nB:", len(B),
            "\nC:", len(C),
            "\nD:", len(D),
            "\nA+B+C+D:", len(A)+len(B)+len(C)+len(D), len(A)+len(B)+len(C)+len(D) == len(gt),
            "\nEoB:", len(EoB), "check:", len(gt) - (len(A) + len(B)),
            "\nEoC:", len(EoC), "check:", len(gt) - (len(A) + len(C)),
            "\nExB:", len(ExB),
            "\nExC:", len(ExC))

        print("\nErrors e1")
        for k,v in e1.items():
            print(k, len(v))

        print("\nErrors e2")
        for k,v in e2.items():
            print(k, len(v))

    return dnd_dict


# =========================================================
# Get results - sequentiall
# =========================================================


def get_dnd_dicts(m1_runs, m2_runs):
    dnd_dicts = {}
    for t in tqdm(TIDE.COCO_THRESHOLDS, desc='Differences'):
        _gt1, _m1, _e1, _d1 = get_dnd_matches(m1_runs[t])
        _gt2, _m2, _e2, _d2 = get_dnd_matches(m2_runs[t])
        assert _gt1 == _gt2
        dnd_dicts[t] = dnd_compare(_gt1, _m1, _m2, _e1, _e2, _d1, _d2)
    return dnd_dicts



def get_dnd_results(ann_file, results_file_1, results_file_2, iou_type):

    gt = TIDE_COCO_DND(ann_file)
    preds_1 = datasets.COCOResult(results_file_1, iou_type)
    preds_2 = datasets.COCOResult(results_file_2, iou_type)

    runs_1 = {}
    runs_2 = {}
    for t in tqdm(TIDE.COCO_THRESHOLDS, desc='Evaluation'):
        tide_1 = TIDE_DND()
        tide_2 = TIDE_DND()

        run_1 = tide_1.evaluate(gt=gt, preds=preds_1, mode='bbox', pos_threshold=t, name=str(t))
        run_2 = tide_2.evaluate(gt=gt, preds=preds_2, mode='bbox', pos_threshold=t, name=str(t))

        runs_1[t] = copy.deepcopy(run_1)
        runs_2[t] = copy.deepcopy(run_2)

    dnd_results = get_dnd_dicts(runs_1, runs_2)

    return dnd_results


# =========================================================
# Get results - multiprocessing
# =========================================================
# TODO: make this faster and activate

# def _get_dnd_dict(m1_run, m2_run):

#     _gt1, _m1, _e1, _d1 = get_dnd_matches(m1_run)
#     _gt2, _m2, _e2, _d2 = get_dnd_matches(m2_run)
#     assert _gt1 == _gt2
#     dnd_dict = dnd_compare(_gt1, _m1, _m2, _e1, _e2, _d1, _d2)

#     return dnd_dict


# def _get_dnd_result(ann_file, results_file_1, results_file_2, iou_type, threshold):

#     gt = TIDE_COCO_DND(ann_file)
#     preds_1 = datasets.COCOResult(results_file_1, iou_type)
#     preds_2 = datasets.COCOResult(results_file_2, iou_type)
#     tide_1 = TIDE_DND()
#     tide_2 = TIDE_DND()
#     run_1 = tide_1.evaluate(gt=gt, preds=preds_1, mode='bbox', pos_threshold=threshold, name=str(threshold))
#     run_2 = tide_2.evaluate(gt=gt, preds=preds_2, mode='bbox', pos_threshold=threshold, name=str(threshold))

#     dnd_result = {threshold: _get_dnd_dict(run_1, run_2)}

#     return dnd_result

# def _get_dnd_result_kwargs(kwargs_dict):
#     return _get_dnd_result(**kwargs_dict)

# import multiprocessing as mp

# def get_dnd_results_mp(ann_file, results_file_1, results_file_2, iou_type):

#     kwargs = [
#         dict(
#             ann_file=ann_file,
#             results_file_1=results_file_1,
#             results_file_2=results_file_2,
#             iou_type=iou_type,
#             threshold=t)
#             for t in TIDE.COCO_THRESHOLDS
#         ]

#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(_get_dnd_result_kwargs, kwargs)

#     return results
