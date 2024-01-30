#! -*- coding : utf-8
import torch

__all__ = ["AUC"]


class AUC(torch.nn.Module):
    def __init__(self, reduction: str = "mean", eps=1e-8):
        super(AUC, self).__init__()
        self.__reduction__ = reduction
        self.__eps__ = eps

    @property
    def reduction(self): return self.__reduction__
    @property
    def eps(self): return self.__eps__

    def forward(self, predicts, truths):
        # TODO expand multi label ditection
        assert truths.dim() == 2, "truths=%d" % (truths.dim())
        assert predicts.dim() == 2, "predicts=%d" % (predicts.dim())
        # assert truths.dim() <= 2 and predicts.dim() <= 2
        # truths = truths.squeeze()
        # predicts = predicts.squeeze()
        # assert truths.dim() == predicts.dim()

        # 特徴量次元とデータ次元を入れ替える
        permute = list(range(truths.dim()))
        permute[0] = truths.dim() - 1
        permute[-1] = 0
        # permute = truths.permute(permute)

        # 全データ間の順序を計算
        # shape = batch x batch
        # 対角 0.5, 比較対象のデータ（0次元）に対して小さいほど0に近づき、大きいほど1に近づく（sigmoid関数をかましているため）
        tmp = predicts.exp()
        scores = tmp / (tmp + tmp.t())  # AUC SCORE: sigmoidの展開式。 同値であれば0.5
        scores.log_()  # logをかけて、勾配を強調
        scores.mul_(-1)  # マイナスをかけてロス化
        del tmp

        # 正例-不例間のマスクを作成
        tp_mask = truths * (1-truths).t()
        scores.mul_(tp_mask)  # 正例－不例間以外のlossを0にする

        loss = scores.sum()
        if self.reduction == "mean":
            loss.div_(tp_mask.sum() + self.eps)
        return loss
