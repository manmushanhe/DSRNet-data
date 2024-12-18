import argparse
import copy
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_enh_s2t_model import ESPnetEnhS2TModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr import decoder_choices as asr_decoder_choices_
from espnet2.tasks.asr import encoder_choices as asr_encoder_choices_
from espnet2.tasks.asr import frontend_choices, normalize_choices
from espnet2.tasks.asr import postencoder_choices as asr_postencoder_choices_
from espnet2.tasks.asr import preencoder_choices as asr_preencoder_choices_
from espnet2.tasks.asr import specaug_choices
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh import decoder_choices as enh_decoder_choices_
from espnet2.tasks.enh import encoder_choices as enh_encoder_choices_
from espnet2.tasks.enh import separator_choices as enh_separator_choices_
from espnet2.tasks.st import STTask
from espnet2.tasks.st import decoder_choices as st_decoder_choices_
from espnet2.tasks.st import encoder_choices as st_encoder_choices_
from espnet2.tasks.st import extra_asr_decoder_choices as st_extra_asr_decoder_choices_
from espnet2.tasks.st import extra_mt_decoder_choices as st_extra_mt_decoder_choices_
from espnet2.tasks.st import postencoder_choices as st_postencoder_choices_
from espnet2.tasks.st import preencoder_choices as st_preencoder_choices_
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    CommonPreprocessor_multi,
    MutliTokenizerCommonPreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# Enhancement
enh_encoder_choices = copy.deepcopy(enh_encoder_choices_)
enh_encoder_choices.name = "enh_encoder"
enh_decoder_choices = copy.deepcopy(enh_decoder_choices_)
enh_decoder_choices.name = "enh_decoder"
enh_separator_choices = copy.deepcopy(enh_separator_choices_)
enh_separator_choices.name = "enh_separator"

# ASR (also SLU)
asr_preencoder_choices = copy.deepcopy(asr_preencoder_choices_)
asr_preencoder_choices.name = "asr_preencoder"
asr_encoder_choices = copy.deepcopy(asr_encoder_choices_)
asr_encoder_choices.name = "asr_encoder"
asr_postencoder_choices = copy.deepcopy(asr_postencoder_choices_)
asr_postencoder_choices.name = "asr_postencoder"
asr_decoder_choices = copy.deepcopy(asr_decoder_choices_)
asr_decoder_choices.name = "asr_decoder"

# ST
st_preencoder_choices = copy.deepcopy(st_preencoder_choices_)
st_preencoder_choices.name = "st_preencoder"
st_encoder_choices = copy.deepcopy(st_encoder_choices_)
st_encoder_choices.name = "st_encoder"
st_postencoder_choices = copy.deepcopy(st_postencoder_choices_)
st_postencoder_choices.name = "st_postencoder"
st_decoder_choices = copy.deepcopy(st_decoder_choices_)
st_decoder_choices.name = "st_decoder"
st_extra_asr_decoder_choices = copy.deepcopy(st_extra_asr_decoder_choices_)
st_extra_asr_decoder_choices.name = "st_extra_asr_decoder"
st_extra_mt_decoder_choices = copy.deepcopy(st_extra_mt_decoder_choices_)
st_extra_mt_decoder_choices.name = "st_extra_mt_decoder"

MAX_REFERENCE_NUM = 100

name2task = dict(
    enh=EnhancementTask,
    asr=ASRTask,
    st=STTask,
)

# More can be added to the following attributes
enh_attributes = [
    "encoder",
    "encoder_conf",
    "separator",
    "separator_conf",
    "decoder",
    "decoder_conf",
    "criterions",
]

asr_attributes = [
    "token_list",
    "input_size",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "preencoder",
    "preencoder_conf",
    "encoder",
    "encoder_conf",
    "postencoder",
    "postencoder_conf",
    "decoder",
    "decoder_conf",
    "ctc_conf",
]

st_attributes = [
    "token_list",
    "src_token_list",
    "input_size",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "preencoder",
    "preencoder_conf",
    "encoder",
    "encoder_conf",
    "postencoder",
    "postencoder_conf",
    "decoder",
    "decoder_conf",
    "ctc_conf",
    "extra_asr_decoder",
    "extra_asr_decoder_conf",
    "extra_mt_decoder",
    "extra_mt_decoder_conf",
]


class EnhS2TTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --enh_encoder and --enh_encoder_conf
        enh_encoder_choices,
        # --enh_separator and --enh_separator_conf
        enh_separator_choices,
        # --enh_decoder and --enh_decoder_conf
        enh_decoder_choices,
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --asr_preencoder and --asr_preencoder_conf
        asr_preencoder_choices,
        # --asr_encoder and --asr_encoder_conf
        asr_encoder_choices,
        # --asr_postencoder and --asr_postencoder_conf
        asr_postencoder_choices,
        # --asr_decoder and --asr_decoder_conf
        asr_decoder_choices,
        # --st_preencoder and --st_preencoder_conf
        st_preencoder_choices,
        # --st_encoder and --st_encoder_conf
        st_encoder_choices,
        # --st_postencoder and --st_postencoder_conf
        st_postencoder_choices,
        # --st_decoder and --st_decoder_conf
        st_decoder_choices,
        # --st_extra_asr_decoder and --st_extra_asr_decoder_conf
        st_extra_asr_decoder_choices,
        # --st_extra_mt_decoder and --st_extra_mt_decoder_conf
        st_extra_mt_decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer
    
    # 增加
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--src_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        
        # ctc类的参数
        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        
        # 增强指标
        group.add_argument(
            "--enh_criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )
        
        # 增强模型配置
        group.add_argument(
            "--enh_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for enh submodel class.",
        )
        
        # asr模型配置
        group.add_argument(
            "--asr_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for asr submodel class.",
        )

        group.add_argument(
            "--st_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for st submodel class.",
        )
        
        # 子任务序列
        group.add_argument(
            "--subtask_series",
            type=str,
            nargs="+",
            default=("enh", "asr"),
            choices=["enh", "asr", "st"],
            help="The series of subtasks in the pipeline.",
        )
        
        # 模型配置
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhS2TModel),
            help="The keyword arguments for model class.",
        )
        
        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        group.add_argument(
            "--src_token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        # 文本清洗
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if "st" in args.subtask_series:
                retval = MutliTokenizerCommonPreprocessor(
                    train=train,
                    token_type=[args.token_type, args.src_token_type],
                    token_list=[args.token_list, args.src_token_list],
                    bpemodel=[args.bpemodel, args.src_bpemodel],
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                    rir_apply_prob=args.rir_apply_prob
                    if hasattr(args, "rir_apply_prob")
                    else 1.0,
                    noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                    noise_apply_prob=args.noise_apply_prob
                    if hasattr(args, "noise_apply_prob")
                    else 1.0,
                    noise_db_range=args.noise_db_range
                    if hasattr(args, "noise_db_range")
                    else "13_15",
                    speech_volume_normalize=args.speech_volume_normalize
                    if hasattr(args, "speech_volume_normalize")
                    else None,
                    speech_name="speech",
                    text_name=["text", "src_text"],
                )
            else:
                retval = CommonPreprocessor_multi(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_name=["text"],
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if inference:
            retval = ("speech",)
        elif train:
            # Recognition mode
            retval = ("speech", "text")
            #retval = ("speech", "text", "speech_ref1")
        else:
            retval = ("speech", "text")
            #retval = ("speech", "text", "speech_ref1")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref1"]
        retval += ["speech_ref1"]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["src_text"]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhS2TModel:
        assert check_argument_types()

        # Build submodels in the order of subtask_series
        # 按子任务序列构建子模型
        model_conf = args.model_conf.copy()
        # subtask_series = ("enh", "asr")
        # 枚举子任务序列中的任务
        for _, subtask in enumerate(args.subtask_series):
            # 为每个子任务配置创建一个字典
            subtask_conf = dict(
                init=None, model_conf=eval(f"args.{subtask}_model_conf")
            )
            
            # getattr返回args对象属性值
            # eg: enc_encoder_conf
            for attr in eval(f"{subtask}_attributes"):
                subtask_conf[attr] = (
                    getattr(args, subtask + "_" + attr, None)
                    if getattr(args, subtask + "_" + attr, None) is not None
                    else getattr(args, attr, None)
                )

            if subtask in ["asr", "st"]:
                m_subtask = "s2t"
            elif subtask in ["enh"]:
                m_subtask = subtask
            else:
                raise ValueError(f"{subtask} not supported.")

            logging.info(f"Building {subtask} task model, using config: {subtask_conf}")
            
            # 构建子模型
            model_conf[f"{m_subtask}_model"] = name2task[subtask].build_model(
                argparse.Namespace(**subtask_conf)
            )

        # 8. Build model
        # 创建enh、asr模型对象
        model = ESPnetEnhS2TModel(**model_conf)

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)
        
        # 返回模型
        assert check_return_type(model)
        return model
