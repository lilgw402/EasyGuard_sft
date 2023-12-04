import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score


def get_gandalf_result(path):
    

    with open(path) as f:
        data = json.load(f)

    y_true = []
    y_pred = []
    a = 0
    
    for ind in data:

        # item = data[ind]
        print(ind)
        break
        reason = str(item.get('reject_reasons', ''))
        if '中危禁售' in reason or '高危禁售' in reason:
            a += 1
            y_true.append(1)
        else:
            y_true.append(0)
        
        y_pred.append(item['gandalf_hit'])

    print(a)
    labels = ['通过', '禁售']
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print('二分类')
    print(report)



def get_gandalf_result_fix(path):
    
    y_true = []
    y_pred = []

    A_C = 0
    B_C = 0
    A, B, C = 0, 0, 0
    alpha = 32

    with open(path) as f:
        data = json.load(f)

        for ind in data:
            item = data[ind]
            reason = item.get('reason', '')
            if '中危禁售' in reason or '高危禁售' in reason:
                y_true = 1
            else:
                y_true = 0

            y_pred = item['gandalf_hit']

            C += y_pred
            B += (1 - y_true)
            A += y_true
            if y_true == 1 and y_pred == 1:
                A_C += 1
            if y_true == 0 and y_pred == 1:
                B_C += 1
            
    precision = A_C / (A_C + alpha * B_C)
    recall = A_C / A
    hit = C / (A + alpha*B)

    print('precision', precision)
    print('recall', recall)
    print('hit', hit)


def get_valley_result(path):

    y_true = []
    y_pred = []
    name = path.split('/')[-1]
    print(name)

    with open(path) as f:
        lines = f.read().split('remark_')
        # print(lines[:10])
        for line in lines:
            if not line: continue
            # print(line)
            line1, predict = line.split('    ')
            id, true_label = line1.split('\t')

            y_true.append(int(true_label))
            product_id = id.split('_')[1] 
            if predict[0] == '否':
                y_pred.append(0)
            elif predict[0] == '是':
                y_pred.append(1)
            else:
                print(predict)

    
    labels = ['通过', '禁售']
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print('二分类')
    print(report)


def get_valley_result_fix(path, src_path):
    data_map = {}
    with open(src_path, 'r') as f:
        for data in f.readlines():
            data = json.loads(data)
            data_map[data['id']] = data['image']
    thresh = 0.5
    print(len(data_map), thresh)

    y_true = []
    y_pred = []

    A_C = 0
    B_C = 0
    A, B, C = 0, 0, 0
    alpha = 32
    name = path.split('/')[-1].split('.')[0]
    print(name)
    pids, gts, preds, images = [], [], [], []

    with open(path) as f:
        lines = f.read().split('remark_')
        # print(lines[:10])
        for line in lines:
            if not line: continue
            # print(line.split('    '))
            try:
                id, true_label, score, predict = line.split('    ')
            except:
                id, true_label, predict = line.split('    ')

            y_true = int(true_label)
            y_pred = 1 if predict[0] == '是' else 0
            # y_pred = 1 if float(score) >= thresh else 0

            C += y_pred
            B += (1 - y_true)
            A += y_true
            if y_true == 1 and y_pred == 1:
                A_C += 1
            if y_true == 0 and y_pred == 1:
                # print(B_C, line)
                pids.append(id.split('_')[0])
                gts.append(true_label)
                preds.append(predict.strip())
                images.append(json.dumps((data_map['remark_' + id])))
                B_C += 1
    
    precision = A_C / (A_C + alpha * B_C)
    recall = A_C / A
    hit = C / (A + alpha*B)

    print(f'AC: {A_C}, BC: {B_C}')
    print('precision', precision)
    print('recall', recall)
    print('hit', hit)

    # df = pd.DataFrame({'product_id': pids, 'gt': gts, 'MLLM_pred': preds, 'images': images})
    # df.to_csv(f'fp_badcase_{name}.csv', index=False)




if __name__ == '__main__': 
    output_path = f'/mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000-debug.txt'
    # output_path = '/mnt/bn/yangmin-priv-fashionmm/Data/zhongheng/jinshou_mllm_output/data-v16-valley-7b-jinshou-class-lora-multi-class-test-45000.txt'
    # output_path = f'/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/valley-7b-v40-{step}-jinshou-class-lora-multi-class-test.txt'
    src_path = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w.txt'
    get_valley_result_fix(output_path, src_path)
    exit()

    for step in range(25000, 50000, 5000):
        output_path = f'/mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v10-valley-7b-jinshou-class-lora-multi-class-test-ocr512-{step}.txt'
        # output_path = '/mnt/bn/yangmin-priv-fashionmm/Data/zhongheng/jinshou_mllm_output/data-v16-valley-7b-jinshou-class-lora-multi-class-test-45000.txt'
        # output_path = f'/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/valley-7b-v40-{step}-jinshou-class-lora-multi-class-test.txt'
        src_path = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w.txt'
        try:
            # get_valley_result(output_path)
            get_valley_result_fix(output_path, src_path)
        except Exception as e:
            print(e)
            continue
    

