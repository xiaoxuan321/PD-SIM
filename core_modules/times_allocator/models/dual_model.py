# -*- coding: utf-8 -*-
"""
双 LSTM 时间模型。

分别预测活动处理时间和等待时间。模型输入由活动嵌入与案例内/案例间
时间特征组成，两个子模型共享序列窗口，但可以使用不同的 Huber delta。
"""

import os

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTM,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    Adagrad,
    Adam,
    Nadam,
    RMSprop,
    SGD,
)

try:
    from support_modules.callbacks import time_callback as tc
except ImportError:
    from importlib import util

    spec = util.spec_from_file_location(
        "time_callback",
        os.path.join(
            os.getcwd(),
            "support_modules",
            "callbacks",
            "time_callback.py",
        ),
    )
    tc = util.module_from_spec(spec)
    spec.loader.exec_module(tc)


def _build_optimizer(parms):
    """根据统一配置创建优化器，确保 learning_rate 真正生效。"""
    optim_name = parms.get("optim", "Nadam")
    learning_rate = float(parms.get("learning_rate", 0.001))

    if optim_name == "Nadam":
        return Nadam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
        )
    if optim_name == "Adam":
        return Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False,
        )
    if optim_name == "RMSprop":
        return RMSprop(
            learning_rate=learning_rate,
            rho=0.9,
        )
    if optim_name == "SGD":
        return SGD(
            learning_rate=learning_rate,
            momentum=0.0,
            nesterov=False,
        )
    if optim_name == "Adagrad":
        return Adagrad(
            learning_rate=learning_rate,
        )

    raise ValueError(
        f"Unsupported optimizer: {optim_name}"
    )


def create_model(
        ac_weights,
        train_vec,
        parms,
        target_type="proc"):
    """
    创建处理时间或等待时间模型。

    Parameters
    ----------
    ac_weights
        预训练活动嵌入矩阵。
    train_vec
        对应子模型的向量化训练数据。
    parms
        TimesModelTrainer 传入的完整配置。
    target_type
        ``proc`` 或 ``wait``，用于选择目标专用 Huber delta。
    """
    if target_type not in {"proc", "wait"}:
        raise ValueError(
            "target_type must be 'proc' or 'wait'"
        )
    if ac_weights is None:
        raise ValueError(
            "Activity embedding weights were not loaded."
        )

    ac_input = Input(
        shape=(train_vec["pref"]["ac_index"].shape[1],),
        name="ac_input",
    )
    features = Input(
        shape=(
            train_vec["pref"]["features"].shape[1],
            train_vec["pref"]["features"].shape[2],
        ),
        name="features",
    )

    ac_embedding = Embedding(
        input_dim=ac_weights.shape[0],
        output_dim=ac_weights.shape[1],
        weights=[ac_weights],
        input_length=train_vec["pref"]["ac_index"].shape[1],
        trainable=bool(
            parms.get("embedding_trainable", False)
        ),
        name="ac_embedding",
    )(ac_input)

    merged = Concatenate(
        name="concatenated",
        axis=2,
    )([ac_embedding, features])

    lstm_units = int(parms["l_size"])
    dropout = float(parms.get("dropout", 0.2))

    sequence_output = LSTM(
        units=lstm_units,
        kernel_initializer="glorot_uniform",
        return_sequences=True,
        dropout=dropout,
        implementation=parms.get("imp", 1),
        name="lstm_sequence",
    )(merged)

    normalized_sequence = BatchNormalization(
        name="sequence_batch_normalization"
    )(sequence_output)

    encoded_output = LSTM(
        units=lstm_units,
        activation=parms.get("lstm_act", "tanh"),
        kernel_initializer="glorot_uniform",
        return_sequences=False,
        dropout=dropout,
        implementation=parms.get("imp", 1),
        name="lstm_encoded",
    )(normalized_sequence)

    times_output = Dense(
        units=train_vec["next"].shape[1],
        activation=parms.get("dense_act", "linear"),
        kernel_initializer="glorot_uniform",
        name="time_output",
    )(encoded_output)

    model = Model(
        inputs=[ac_input, features],
        outputs=[times_output],
        name=f"{target_type}_time_model",
    )

    if target_type == "wait":
        huber_delta = float(
            parms.get(
                "wait_huber_delta",
                parms.get("huber_delta", 1.0),
            )
        )
    else:
        huber_delta = float(
            parms.get(
                "proc_huber_delta",
                parms.get("huber_delta", 1.0),
            )
        )

    model.compile(
        loss={
            "time_output": Huber(
                delta=huber_delta
            )
        },
        optimizer=_build_optimizer(parms),
        metrics={
            "time_output": [
                "mean_absolute_error"
            ]
        },
    )

    print(
        f"[dual_model] target={target_type}, "
        f"n_size={train_vec['pref']['ac_index'].shape[1]}, "
        f"l_size={lstm_units}, dropout={dropout}, "
        f"huber_delta={huber_delta}, "
        f"learning_rate={parms.get('learning_rate', 0.001)}"
    )
    model.summary()
    return model


def _build_callbacks(model_file, parms):
    """为每个子模型创建独立回调，避免跨 fit 共享状态。"""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=int(
                parms.get("early_stopping_patience", 50)
            ),
            restore_best_weights=True,
            verbose=0,
        ),
        ModelCheckpoint(
            model_file,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(
                parms.get("lr_patience", 10)
            ),
            verbose=0,
            mode="min",
            min_delta=0.0001,
            cooldown=0,
            min_lr=float(
                parms.get("min_learning_rate", 1e-6)
            ),
        ),
        tc.TimingCallback(parms["output"]),
    ]


def _training_model(
        ac_weights,
        train_vec,
        valdn_vec,
        parms):
    """训练处理时间和等待时间两个子模型。"""
    print("Build processing-time and waiting-time models...")

    batch_size = int(parms["batch_size"])
    epochs = int(parms["epochs"])
    path = parms["output"]
    fname = parms["file"].split(".")[0]

    if parms["all_r_pool"]:
        proc_model_file = os.path.join(
            path,
            fname + "_dpiapr.h5",
        )
        waiting_model_file = os.path.join(
            path,
            fname + "_dwiapr.h5",
        )
    else:
        proc_model_file = os.path.join(
            path,
            fname + "_dpispr.h5",
        )
        waiting_model_file = os.path.join(
            path,
            fname + "_dwispr.h5",
        )

    proc_model = create_model(
        ac_weights,
        train_vec["proc_model"],
        parms,
        target_type="proc",
    )
    proc_model.fit(
        {
            "ac_input": train_vec[
                "proc_model"
            ]["pref"]["ac_index"],
            "features": train_vec[
                "proc_model"
            ]["pref"]["features"],
        },
        {
            "time_output": train_vec[
                "proc_model"
            ]["next"]
        },
        validation_data=(
            {
                "ac_input": valdn_vec[
                    "proc_model"
                ]["pref"]["ac_index"],
                "features": valdn_vec[
                    "proc_model"
                ]["pref"]["features"],
            },
            {
                "time_output": valdn_vec[
                    "proc_model"
                ]["next"]
            },
        ),
        verbose=2,
        callbacks=_build_callbacks(
            proc_model_file,
            parms,
        ),
        batch_size=batch_size,
        epochs=epochs,
    )

    waiting_model = create_model(
        ac_weights,
        train_vec["waiting_model"],
        parms,
        target_type="wait",
    )
    waiting_model.fit(
        {
            "ac_input": train_vec[
                "waiting_model"
            ]["pref"]["ac_index"],
            "features": train_vec[
                "waiting_model"
            ]["pref"]["features"],
        },
        {
            "time_output": train_vec[
                "waiting_model"
            ]["next"]
        },
        validation_data=(
            {
                "ac_input": valdn_vec[
                    "waiting_model"
                ]["pref"]["ac_index"],
                "features": valdn_vec[
                    "waiting_model"
                ]["pref"]["features"],
            },
            {
                "time_output": valdn_vec[
                    "waiting_model"
                ]["next"]
            },
        ),
        verbose=2,
        callbacks=_build_callbacks(
            waiting_model_file,
            parms,
        ),
        batch_size=batch_size,
        epochs=epochs,
    )

    return {
        "proc_model": {
            "model": proc_model
        },
        "wait_model": {
            "model": waiting_model
        },
    }
