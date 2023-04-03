import json
from tqdm import tqdm
from cruise.utilities.hdfs_io import hlist_files, hopen

idx = 0
skip = 0
cache_limit = 20000
cache = []
files = hlist_files(['hdfs://haruna/home/byte_ecom_govern/user/wangxian/datasets/audio4sv'])
files = [f for f in files if '_SUCCESS' not in f]
for file in tqdm(files):
    with hopen(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            user_id, room_id, snapshot_id, info, *args = str(line, encoding='utf-8').split('\t')
        except Exception as e:
            # print(line)
            skip += 1
            if skip % 100 == 0:
                print(f'skipped: {skip} samples')
            continue
        voice = json.loads(info)
        audio_urls = []
        for audioslice in voice['voice_text']:
            if len(audioslice['text']) > 10:
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
                with open(f'cache/audio_samples_{idx}.jsonl', 'w') as f:
                    f.writelines('\n'.join(cache))
                cache = []
                idx += 1

if cache:
    with open(f'cache/audio_samples_{idx}.jsonl', 'w') as f:
        f.writelines('\n'.join(cache))
    cache = []
    idx += 1