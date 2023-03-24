# GandalfOnCruise
# 一、Gandalf为什么要使用Cruise
1. Cruise针对原生torch在介入公司内部算力(Arnold)，内部数据(kv,parquet)，弹性训练(Elastic Training)，分布式训练(DDP,DDPNPU,DeepSpeed)等方面做了定制优化，让算法同学专注于业务算法开发，减少了其他wet-hands工作的开发量(读取远程数据/针对分布式训练手写梯度更新等)
2. 组内内容理解基建所有成果均托管于EasyGuard，包括多模态预训练，NLP预训练，LLM能力探索等。而EasyGuard的模型能力目前大部分是基于纯粹的Cruise框架训练的，为了快速复用组内现有能力，将现在托管在其他框架的Gandalf整体流程适配Cruise，并托管在EasyGaurd上/Gandalf。这样打通了底层内容能力理解和上层业务应用之间的藩篱。
# 二、如何使用Cruise
目前的Cruise缺乏一个比较简明的上手说明，组内内容理解同学的预训练使用也比较分散，缺乏一个通用/简洁的教程。Cruise使用的sop是全局配置化开发，而Gandalf本身因为业务形态的特点：配置项繁杂，参数众多，需要多人合作，因此在设计之初就专为配置化开发而优化，因此下面借由Gandalf开发（第三节）来演示下如何在Cruise上快速开发模型。
Cruise进阶使用请参考[Cruise 使用必读 | Must Read](https://bytedance.feishu.cn/wiki/wikcndoXvN2g2tuQSF76mebqnwh) 
# 三、接入Gandalf能力SOP
## 0. Gandalf的背景/实验文档/历史总结
* [电商治理Gandalf模型部署/使用文档内容侧甘道夫模型训练部署指北](https://bytedance.feishu.cn/docx/doxcnkpqA80nAT60I4JZDDAbtqA)
* [电商治理Gandalf NN模型部署/使用文档直播甘道夫NN模型训练部署指北](https://bytedance.feishu.cn/docx/doxcnCaGqFJ6ojsE9axvkcuYpDb)
## 1. 前期准备并跑通基于Merlin的训练流程demo
- 对于对于内容侧同学，直接在EasyGaurd上/Gandalfbranch内的ecom_live_gandalf目录下新fork已有的config.yaml并rename，对于对于其他方向同学，在EasyGaurd上/Gandalfbranch内新建对应业务目录并forkecom_live_gandalf内的config.yaml并rename
- fork已有的GandalfTrial并rename，在merlin开发机上launch实验机器作为debug使用
```bash
launch --gpu 2 --cpu 20 --memory 40 -- doas --krb5-username xxx bash
```
运行以下训练入口命令，用于模拟正式训练任务,ecom_live_gandalf修改为自己的业务，live_gandalf_autodis_nn_asr.yaml改为自己修改好的配置)，开发机调试无误后提交代码到gitlab。
```bash
bash examples/gandalf/run_on_merlin.sh --config=config/ecom_live_gandalf/live_gandalf_autodis_nn_asr.yaml --fit
```
## 2. 正式开发
### 2.1 配置config文件
基于yaml文件存储，必须包括以下三个配置，trainer/model/data，trainer对应训练相关参数，model对应模型开发对应参数，data对应数据开发参数，cruise配置的重点在于，配置文件中所有配置项必须在对应的模块有着对应的配置项， 否则会报错，如果报了类似于argument/cli等error，仔细检查下参数映射能否一一对上。
### 2.2 trainer训练器开发
默认只需要修改以下参数：
      1. default_root_dir： bytednas上的本地模型训练产物地址  
      2. default_hdfs_dir： hdfs上的模型训练产物地址  
      3. project_name： tracking项目名称，默认gandalf  
      4. experiment_name: tracking实验名称，自定义  
### 2.3 CruiseDatamodule数据开发
    1. 目前已经针对CruiseDataModule做了二次封装，对于batch transform等实现无需操心，只需要关注核心的特征处理和label组装部分。
    2. dataset
      1. input_dir: 基于hdfs的训练数据目录
      2. val_input_dir: 基于hdfs的验证数据目录
      3. test_input_dir: 基于hdfs的测试数据目录
      4. train_folder: 训练数据周期
      5. val_folder: 验证数据周期
      6. test_folder: 测试数据周期
    3. feature_provider：
      1. type：指定特征预处理类 详情参考gandalf/dataset/custom_dataset/ecom_live_gandalf/EcomLiveGandalfParquetAutoDisCruiseDataModule
      2. 其他均为特征处理特异参数
    4. data_factory：
      1. batch_size： batch size
    5. Type: 指定datamodule 详情参考gandalf/dataset/custom_dataset/ecom_live_gandalf/EcomLiveGandalfParquetAutoDisCruiseDataModule
### 2.4 CruiseModule模型开发
    Model
      1. type：指定模型类 详情参考gandalf/models/custom_models/ecom_live_gandalf/ecom_live_gandalf_auto_dis_nn_asr_cruise_model
      2. 其他均为模型特异参数
# 3. 在merlin提交正式任务训练
观察对应tracking页面的Gandalf指标(Auc/Precision/Recall/loss/avg_output)等
