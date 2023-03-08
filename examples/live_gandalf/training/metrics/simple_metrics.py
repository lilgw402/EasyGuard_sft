# coding=utf-8
# Email: panziqi@bytedance.com
# Create: 2021/9/20 11:42 上午

from utils.registry import METRICS
from training.metrics.base_metric import BaseMetric
from typing import List, Union
from utils.analyzer import roc_curve,mae_sklearn,mse_sklearn,rmse_sklearn,pr_score_analysis,precision_rescall_score


class SimpleMetric(BaseMetric):
    def __init__(self, score_key, label_key, score_idx=1):
        self._score_key = score_key
        self._label_key = label_key
        self._score_idx = score_idx

    def collect_pre_labels(self, result):
        probs = []
        labels = []
        for res in result:
            prob = res[self._score_key]
            if isinstance(prob, list):
                if isinstance(self._score_idx, list):
                    prob = sum([prob[i] for i in self._score_idx])
                    label = res[self._label_key]
                else:
                    prob = prob[self._score_idx]
                    label = res[self._label_key] == self._score_idx
            else:
                label = res[self._label_key] == self._score_idx
            probs.append(prob)
            labels.append(label)
        return probs, labels

    def cal_metric(self, scores, labels):
        raise NotImplementedError


@METRICS.register_module()
class AUC(SimpleMetric):
    def __init__(
        self,
        score_key: str,
        label_key: str,
        show_threshold_details: bool = True,
        score_idx: Union[int, List] = 1,
        pr_scope: List[int] = None,
        fix_recall: float = 0.3,
        dump_table_name: str = None,
        hdfs_table_name: str = None
    ):
        super().__init__(score_key, label_key, score_idx)
        self._show_threshold_details = show_threshold_details
        self._pr_scope = [0, 1, 200] if pr_scope is None else pr_scope
        self._fix_recall = fix_recall
        self._dump_table_name = dump_table_name
        self._hdfs_table_name = hdfs_table_name

    def cal_metric(self, scores, labels):
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        result = {
            f"{self._score_key}-AUC": round(roc_curve(scores, labels), 3),
            "Total": len(labels),
            "PosCnt": positive_count,
            "NegCnt": negative_count,
            "PosRate": round(
                positive_count / (negative_count + positive_count + 1e-12), 3
            ),
        }
        if self._show_threshold_details:
            fix_recall_metric_dict = pr_score_analysis(
                scores,
                labels,
                start=self._pr_scope[0],
                end=self._pr_scope[1],
                partition=self._pr_scope[2],
                fixed_recall=self._fix_recall,
                dump_table_name=self._dump_table_name,
                hdfs_table_name=self._hdfs_table_name
            )
            result.update(fix_recall_metric_dict)
            table_name = self._dump_table_name.replace(".csv", f"_{self.name}.csv") if isinstance(self._dump_table_name,
                                                                                                  str) else None
            pr_score_analysis(
                scores,
                labels,
                start=self._pr_scope[0],
                end=self._pr_scope[1],
                partition=self._pr_scope[2],
                dump_table_name=table_name,
            )

        return result


@METRICS.register_module()
class MSE(SimpleMetric):
    def __init__(self, score_key, label_key):
        super().__init__(score_key, label_key)

    def cal_metric(self, scores, labels):
        result = {
            f"{self._score_key}-MSE": mse_sklearn(scores, labels),
        }
        return result


@METRICS.register_module()
class MAE(SimpleMetric):
    def __init__(self, score_key, label_key):
        super().__init__(score_key, label_key)

    def cal_metric(self, scores, labels):
        result = {
            f"{self._score_key}-MAE": mae_sklearn(scores, labels),
        }
        return result


@METRICS.register_module()
class RMSE(SimpleMetric):
    def __init__(self, score_key, label_key):
        super().__init__(score_key, label_key)

    def cal_metric(self, scores, labels):
        result = {
            f"{self._score_key}-RMSE": rmse_sklearn(scores, labels),
        }
        return result


@METRICS.register_module()
class MeanPrecisionRecall(SimpleMetric):
    def __init__(self, score_key, label_key, shares=[0.1], score_idx=1, round_result=True):
        super().__init__(score_key, label_key, score_idx)
        if not isinstance(shares, list):
            shares = list(shares)
        self._shares = shares
        if round_result:
            self._round = round
        else:
            self._round = lambda x, y: x

    def cal_metric(self, scores, labels):
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        result = {
            "Total": len(labels),
            "PosCnt": positive_count,
            "NegCnt": negative_count,
            "Pos/Neg Ratio": round(positive_count / (negative_count + 1e-12), 3),
        }
        scores_sorted = list(sorted(scores, reverse=True))
        precisions, recalls = [], []
        for share in self._shares:
            threshold = scores_sorted[int(share * len(scores))]
            precision, recall, review_rate, _, _, _, _ = precision_rescall_score(
                scores, labels, threshold=threshold
            )
            precisions.append(precision)
            recalls.append(recall)
            result.update({f"{self._score_key}-recall-{share}": self._round(recall, 4)})
            result.update({f"{self._score_key}-precision-{share}": self._round(precision, 4)})
        result.update({f"{self._score_key}-recall-mean": self._round(sum(recalls) / len(recalls), 4)})
        result.update({f"{self._score_key}-precision-mean": self._round(sum(precisions) / len(precisions), 4)})
        return result


@METRICS.register_module()
class MPR(MeanPrecisionRecall):
    def __init__(self, score_key, label_key, shares=[0.1], score_idx=1, round_result=True):
        super(MPR, self).__init__(score_key, label_key, shares=shares, score_idx=score_idx, round_result=round_result)