import json

# data_dir = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/train_data/train_data_v1.json'
# data_dir = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/train_data/train_data_v5.json'
data_dir = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/train_data/train_data_v12.json'
# data_dir = '/mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n.txt'

if data_dir[-4:] != 'json':
    data = [json.loads(data) for data in open(data_dir, 'r').readlines()]
else:
    data = json.load(open(data_dir, "r"))
print(f'data size: {len(data)}')
print(data[0])

for d in data:
    prompt = d['conversations'][0]['value']
    image_num = len(d['image'])
    sp_tokens = ''
    for i in range(image_num):
        sp_tokens += f'<image{i}>'
    d['conversations'][0]['value'] = prompt.replace('<video>', sp_tokens)


res = json.dumps(data, ensure_ascii=False)
with open((data_dir.split('.')[0] + '_valley_product.json'), 'w') as file:
    file.write(res)

