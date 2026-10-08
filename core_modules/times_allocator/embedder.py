from core_modules.times_allocator import embedding_trainer as et
from core_modules.times_allocator import embedding_trainer_act_weighting_no_times as etawnt
from core_modules.times_allocator import embedding_trainer_act_weighting_times as etawt
from core_modules.times_allocator import embedding_trainer_times as ett
from core_modules.times_allocator import embedding_word2vec as ew
from support_modules.common import EmbeddingMethods as Em


class Embedder:
    """
    嵌入矩阵生成器类，负责根据不同的嵌入方法训练和生成活动/用户的嵌入向量。
    主要功能：
    - 根据配置选择具体的嵌入算法（如 Word2Vec、点积等）
    - 训练嵌入模型并生成向量表示
    - 保存嵌入矩阵文件到指定路径
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        """构造函数，初始化嵌入生成器

        参数:
            params: 参数字典，包含：
                - file: 日志文件名
                - embedded_path: 嵌入矩阵保存路径
                - include_times: 是否包含时间信息
            log: 事件日志数据
            ac_index: 活动名称到索引的映射字典
            index_ac: 索引到活动名称的映射字典
            usr_index: 用户名称到索引的映射字典
            index_usr: 索引到用户名称的映射字典
        """
        self.log = log.copy()  # 日志数据副本
        self.params = params  # 参数字典
        print(params)  # 打印参数（调试用）

        # 索引映射
        self.ac_index = ac_index  # 活动名称 -> 索引
        self.index_ac = index_ac  # 索引 -> 活动名称
        self.usr_index = usr_index  # 用户名称 -> 索引
        self.index_usr = index_usr  # 索引 -> 用户名称

        # 从参数中提取关键配置
        self.file_name = params['file']  # 输入日志文件名（如 ac_DP_cvs_pharmacy.csv）
        self.embedded_path = params['embedded_path']  # 嵌入矩阵保存目录（如 input_files/embedded_matrix/）
        self.include_times = params['include_times']  # 是否在嵌入中包含时间特征

        # 嵌入文件名（将在 create_embeddings 方法中赋值）
        self.embedding_file_name = None

    def create_embeddings(self, method):
        """
        核心方法：创建嵌入矩阵并保存文件

        参数:
            method: 嵌入方法（如 Em.W2VEC 表示 Word2Vec）

        返回:
            嵌入矩阵（numpy.ndarray 或类似结构）

        注意：
            - 实际文件保存操作发生在具体嵌入器类（如 EmbeddingWord2vec）中
            - 生成的文件路径会保存在 self.embedding_file_name
        """
        # 1. 根据方法名称获取对应的嵌入器类
        embedder_class = self._get_embedder(method)

        # 2. 初始化具体的嵌入器（此时会触发文件生成）
        embedder = embedder_class(
            self.params,
            self.log,
            self.ac_index,
            self.index_ac,
            self.usr_index,
            self.index_usr
        )

        # 3. 记录嵌入文件名（由具体嵌入器生成）
        self.embedding_file_name = embedder.embedding_file_name

        # 4. 加载嵌入矩阵并返回
        return embedder.load_embeddings()

    def _get_embedder(self, method):
        """
        工厂方法：根据嵌入方法返回对应的嵌入器类

        参数:
            method: 嵌入方法枚举值（如 Em.W2VEC）

        返回:
            具体的嵌入器类（如 EmbeddingWord2vec）

        注意：
            - 不同嵌入器会在初始化时生成嵌入文件
            - 文件通常保存在 self.embedded_path 目录下
        """
        if method == Em.DOT_PROD:
            return et.EmbeddingTrainer  # 基础点积嵌入
        elif method == Em.W2VEC:
            return ew.EmbeddingWord2vec  # Word2Vec 嵌入（此时会生成 .emb 文件）
        elif method == Em.DOT_PROD_TIMES:
            return ett.EmbeddingTrainer  # 含时间特征的点积嵌入
        elif method == Em.DOT_PROD_ACT_WEIGHT and self.include_times:
            return etawt.EmbeddingTrainer  # 带活动权重的点积嵌入（含时间）
        elif method == Em.DOT_PROD_ACT_WEIGHT and not self.include_times:
            return etawnt.EmbeddingTrainer  # 带活动权重的点积嵌入（不含时间）
        else:
            raise ValueError(f"不支持的嵌入方法: {method}")