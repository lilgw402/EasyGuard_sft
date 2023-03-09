from transformers import pipeline

generator = pipeline('text-generation', model="bigscience/bloom-3b")

inputs = [
    '周杰伦',
    '世界上最高的山峰',
    '小炒肉怎么做',
    '怎么成功',
    '喝茶水对身体有什么好处和坏处',
    '冰箱什么品牌最好排名前十名',
    '说一说生活中哪些物品是用金属做成的',
    '写一首关于风花雪月的诗',
    '烟草局是干什么的',
    '1k和2k显示屏区别大吗',
    '经期快结束了可以拔罐吗',
    'dns作用',
    '嗓子有血痰是怎么回事',
    '早晨喝牛奶吃鸡蛋好不好',
    '爸爸的哥哥叫什么',
    'b站视频怎么下载到电脑',
    '一个人的情感一直不顺是怎么回事',
]

for input in inputs:
    input = input + '?'
    results = generator(input, max_length=60)
    print(f'input: {input}, output: {results}')
