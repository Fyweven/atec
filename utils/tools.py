# -*- coding: utf-8 -*-

def getF1score(pred, label):
    total_count = 0
    for b in range(batch_size):
        total_count += 1
        t = sim_label.data[b]
        v = output.data[b]
        t = int(t)
        v = (1 if v > 0.5 else 0)
        predict_count[v] += 1
        target_count[t] += 1
        if t == v:
            right_count[t] += 1


p = (right_count[1] / predict_count[1] if predict_count[1] > 0 else 0.0)
r = (right_count[1] / target_count[1] if target_count[1] > 0 else 0.0)
a = (right_count[0] + right_count[1]) / total_count
f1 = (2 * p * r / (p + r) if p + r > 0 else 0.0)
print('loss: %.4f' % run_loss)
print('right: ' + str(right_count))
print('target: ' + str(target_count))
print('predict: ' + str(predict_count))
print('accuracy: %.4f, F1 score: %.4f' % (a, f1))