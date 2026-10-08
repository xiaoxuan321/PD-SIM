import math
import random

from support_modules.common import EmbeddingMethods as Em  # 导入自定义的嵌入方法工具
import gensim  # 用于词向量训练的库
import numpy as np  # 数值计算库


class EmbeddingBase:
    """基础嵌入类，用于学习活动和用户的嵌入表示"""

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        """
        初始化嵌入类

        参数:
            params: 配置参数字典，包含文件路径、嵌入方法等信息
            log: 事件日志数据
            ac_index: 活动名称到索引的映射字典
            index_ac: 索引到活动名称的映射字典
            usr_index: 用户名到索引的映射字典
            index_usr: 索引到用户名的映射字典
        """
        self.ac_weights = []  # 存储活动权重（如有）
        self.log = log.copy()  # 事件日志副本
        self.ac_index = ac_index  # 活动名称->索引字典
        self.index_ac = index_ac  # 索引->活动名称字典
        self.usr_index = usr_index  # 用户名->索引字典
        self.index_usr = index_usr  # 索引->用户名字典

        # 文件路径相关设置
        self.file_name = params['file']  # 原始日志文件名
        self.embedded_path = params['embedded_path']  # 嵌入文件的存储路径

        # 生成嵌入矩阵和模型的文件名
        self.embedding_file_name = Em.get_matrix_file_name(
            params['emb_method'], params['include_times'], params['concat_method'], self.file_name)
        self.embedding_model_file_name = Em.get_model_file_name(
            params['emb_method'], params['include_times'], self.file_name)

    def learn_characteristics(self, sent, vector_size, characteristic):
        """
        使用FastText学习活动或用户的嵌入表示

        参数:
            sent: 训练语料，格式为[[word1, word2,...], ...]
            vector_size: 嵌入向量的维度
            characteristic: 要学习的特征类型 ('activities' 或 'users')

        返回:
            char_dict: 包含特征及其嵌入向量的字典
        """
        # 创建并训练FastText模型
        # min_count=0 表示训练所有词汇，window=6 表示上下文窗口大小为6
        model = gensim.models.FastText(sent, vector_size=vector_size, min_count=0, window=6)

        # 训练模型100个epoch
        nr_epochs = 100
        for epoch in range(nr_epochs):
            if epoch % 20 == 0:
                print('Now training epoch %s word2vec' % epoch)

            # 训练模型并动态调整学习率
            model.train(sent, start_alpha=0.025, epochs=nr_epochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # 降低学习率
            model.min_alpha = model.alpha  # 固定学习率，不再衰减

        print(model.wv.key_to_index)  # 打印词汇索引

        # 根据特征类型构建嵌入字典
        if characteristic == 'activities':
            # 按索引排序活动名称
            keys_sorted = [x[0] for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]
            # 创建活动嵌入字典
            char_dict = {}
            for key in keys_sorted:
                char_dict[key] = model.wv[key]
        else:
            # 对于用户特征，获取所有唯一词汇
            unique_sent = list(set([item for sublist in sent for item in sublist]))
            # 创建用户嵌入字典
            char_dict = {x: model.wv[x] for x in unique_sent}

        return char_dict

    def vectorize_input(self, log, negative_ratio=1.0):
        """
        生成用于嵌入学习的正负样本对

        参数:
            log: 事件日志数据
            negative_ratio: 负样本与正样本的比例

        返回:
            包含特征索引的字典和对应的标签
        """
        # 创建活动-用户对列表 (正样本)
        pairs = list()
        for i in range(0, len(self.log)):
            # 记录每个事件的活动和用户对
            pairs.append((self.ac_index[self.log.iloc[i]['task']],
                          self.usr_index[self.log.iloc[i]['user']]))

        # 计算正样本数量（取日志长度的一半）
        n_positive = math.ceil(len(self.log) / 2)
        # 计算批次大小（正样本+负样本）
        batch_size = int(n_positive * (1 + negative_ratio))
        # 初始化批次数组 [活动索引, 用户索引, 标签]
        batch = np.zeros((batch_size, 3))
        pairs_set = set(pairs)  # 转为集合便于快速查找

        # 获取所有活动和用户名称
        activities = list(self.ac_index.keys())
        users = list(self.usr_index.keys())

        # 随机选择正样本
        idx = 0
        for idx, (activity, user) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (activity, user, 1)  # 正样本标签为1

        # 增加索引
        idx += 1

        # 添加负样本直到达到批次大小
        while idx < batch_size:
            # 随机选择活动和用户
            random_ac = random.randrange(len(activities) - 1)
            random_rl = random.randrange(len(users) - 1)

            # 确保不是正样本
            if (random_ac, random_rl) not in pairs_set:
                # 添加负样本，标签为0
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # 打乱样本顺序
        np.random.shuffle(batch)

        # 返回特征和标签
        return {'activity': batch[:, 0], 'user': batch[:, 1]}, batch[:, 2]