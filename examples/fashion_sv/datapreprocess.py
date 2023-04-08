import json
import time
from tqdm import tqdm
from cruise.utilities.hdfs_io import hlist_files, hopen

idx = 0
skip = 0
cache_dir = './cache'  # f'/mnt/bd/wx-nas/lost+found/audiosamples'
cache_limit = 20000
cache = []
files = hlist_files(['hdfs://haruna/home/byte_ecom_govern/user/wangxian/datasets/fashionaudio/audio4sv/raw_0403_0405'])
files = [f for f in files if '_SUCCESS' not in f]
# files = ['audio_parts']
for file in tqdm(files):
    with hopen(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            if isinstance(line, str):
                dline = line
            elif isinstance(line, bytes):
                dline = str(line, encoding='utf-8')
            user_id, room_id, snapshot_id, info, *args = dline.split('\t')
        except Exception as e:
            print('error occurred:', type(line), e)
            skip += 1
            if skip % 100 == 0:
                print(f'skipped: {skip} samples')
            continue
        voice = json.loads(info)
        audio_urls = []
        for audioslice in voice['voice_text']:
            if len(audioslice['text']) > 20:
                audio_urls.append(audioslice['audio_url'])
        if audio_urls:
            sample = {
                'user_id': user_id,
                'room_id': room_id,
                'snapshot_id': snapshot_id,
                'audio_urls': audio_urls
            }
            cache.append(json.dumps(sample))

            if len(cache) >= cache_limit:
                with open(f'{cache_dir}/audio_samples_{idx}.jsonl', 'w') as f:
                    f.writelines('\n'.join(cache))
                cache = []
                idx += 1

if cache:
    with open(f'{cache_dir}/audio_samples_{idx}.jsonl', 'w') as f:
        f.writelines('\n'.join(cache))
    cache = []
    idx += 1
