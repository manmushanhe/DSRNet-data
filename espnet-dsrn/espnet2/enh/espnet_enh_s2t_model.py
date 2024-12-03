import logging
import random
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.st.espnet_model import ESPnetSTModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.layers.stft import Stft
from espnet2.layers.log_mel import LogMel
from torch_complex.tensor import ComplexTensor


if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

# 构造函数包含enh模型和asr模型
class ESPnetEnhS2TModel(AbsESPnetModel):
    """Joint model Enhancement and Speech to Text."""
    
    # 构造方法，在创建实例化对象时执行
    def __init__(
        self,
        enh_model: ESPnetEnhancementModel,
        s2t_model: Union[ESPnetASRModel, ESPnetSTModel],
        calc_enh_loss: bool = True,
        weight_enh: int = 300,
        weight_refine: int = 100,
        use_crnn: bool = False,
        bypass_enh_prob: float = 0,  # 0 means do not bypass enhancement for all data
    ):
        assert check_argument_types()

        super().__init__()
        if use_crnn:
            self.stft = Stft(
                n_fft=320,
                win_length=320,
                hop_length=128,
                center=True,
                window="hann",
                normalized=False,
                onesided=True,
                )
        else:
            self.stft = Stft(
                n_fft=512,
                win_length=512,
                hop_length=128,
                center=True,
                window="hann",
                normalized=False,
                onesided=True,
                )
        # 增强模型
        self.enh_model = enh_model
        if use_crnn:
            self.logmel = LogMel(
                fs=16000,
                n_fft=320,
                n_mels=80,
                fmin=None,
                fmax=None,
                htk=False,
                )
        else:
            self.logmel = LogMel(
                fs=16000,
                n_fft=512,
                n_mels=80,
                fmin=None,
                fmax=None,
                htk=False,
                )
        '''
        elif self.enh_model.use_concate:
            self.logmel = LogMel(
                fs=16000,
                n_fft=1024,
                n_mels=80,
                fmin=None,
                fmax=None,
                htk=False,
                )
        '''
        
        # asr模型
        self.s2t_model = s2t_model  # ASR or ST model
        self.weight_enh = weight_enh
        self.weight_refine = weight_refine

        self.calc_enh_loss = calc_enh_loss
        self.extract_feats_in_collect_stats = (
            self.s2t_model.extract_feats_in_collect_stats
        )
       
    # speech是带噪语   
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech_ref1: torch.Tensor = None,
        speech_ref1_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)


        batch_size = speech.shape[0]

        if speech_ref1 is not None:
            enhloss_flag = True
        else:
            enhloss_flag = False
        if self.enh_model.cascaded:
            enhloss_flag = False

        loss_enh = None 
        # 1. Enhancement
        # model forward
        if not self.enh_model.use_demucs:
            feats, feats_lens = self._compute_stft(speech, speech_lengths)
            feats_spectrogram = feats.real**2 + feats.imag**2
            feats_magnitude = torch.sqrt(feats_spectrogram)

        if self.enh_model.use_demucs:
            pred_speech = self.enh_model.forward_enhance(speech, speech_lengths)
            pred_feats, feats_lens = self._compute_stft(pred_speech, speech_lengths)
            pred_feats_spectrogram = pred_feats.real**2 + pred_feats.imag**2
            pred_feats_magnitude = torch.sqrt(pred_feats_spectrogram)
            pred_mask = None
            feats_magnitude = None
        elif self.enh_model.use_crnn:
            pred_feats_magnitude = self.enh_model.forward_enhance(feats_magnitude, feats_lens)
            pred_mask = None
        elif self.enh_model.use_dcunet:
            enhanced_speech, loss_enh = self.enh_model.forward_dcunet(speech)
            pred_feats, feats_lens = self._compute_stft(enhanced_speech, speech_lengths)
            #feats_lens = ( speech_lengths - 1 ) // self.enh_model.encoder.hop_length + 1 
            feats_spectrogram = feats.real**2 + feats.imag**2
            feats_magnitude = torch.sqrt(feats_spectrogram)
            pred_feats_magnitude = feats_magnitude
        else:    
            pred_mask = self.enh_model.forward_enhance(feats_magnitude, feats_lens)
            pred_feats_magnitude = pred_mask*feats_magnitude
        

        # compute loss
        # 计算增强后损失函数   
            
        if enhloss_flag:
            if not self.enh_model.skip_cal_loss and not self.enh_model.cal_dcunet_loss:
                feats_ref1, feats_ref1_lens = self._compute_stft(speech_ref1, speech_ref1_lengths)
                feats_ref1_spectrogram = feats_ref1.real**2 + feats_ref1.imag**2
                feats_ref1_magnitude = torch.sqrt(feats_ref1_spectrogram)
                loss_enh, _, _ = self.enh_model.forward_loss(
                    feats_ref1_magnitude,
                    feats_magnitude,
                    pred_feats_magnitude,
                    pred_mask,
                )         
                loss_enh = loss_enh[0]

        loss_dsrn = None
        # dsrn moudle
        if self.enh_model.use_dsrn:
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude, pre_noise_magnitude = self.enh_model.dsrn(enhanced=pred_feats_magnitude, noise=feature_noise_pre)
            if not self.enh_model.skip_refine_loss:
                loss_dsrn = self.enh_model.dsrn.forward_loss(pred_feats_magnitude, pre_noise_magnitude, feats_ref1_magnitude, feats_magnitude)     
        
        if self.enh_model.use_dsrn_linear_mask:
            pred_mask, pred_noise_mask = self.enh_model.dsrn(enhanced=pred_mask, noise=1 - pred_mask)
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude = pred_mask * feats_magnitude              #pred_feats_magnitude = pred_mask * pred_feats_magnitude
            pre_noise_magnitude =  pred_noise_mask * feats_magnitude        #pre_noise_magnitude =  pred_noise_mask * feature_noise_pre
            if not self.enh_model.skip_refine_loss:
                loss_dsrn = self.enh_model.dsrn.forward_loss(pred_feats_magnitude, pre_noise_magnitude, feats_ref1_magnitude, feats_magnitude)     
        
        if self.enh_model.use_dsrn_conv2d_mask:
            pred_mask, pred_noise_mask = self.enh_model.dsrn(enhanced=pred_mask, noise=1 - pred_mask)
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude = pred_mask * feats_magnitude              #pred_feats_magnitude = pred_mask * pred_feats_magnitude
            pre_noise_magnitude =  pred_noise_mask * feats_magnitude        #pre_noise_magnitude =  pred_noise_mask * feature_noise_pre

            if not self.enh_model.skip_refine_loss:
                loss_dsrn = self.enh_model.dsrn.forward_loss(pred_feats_magnitude, pre_noise_magnitude, feats_ref1_magnitude, feats_magnitude)     
        
        if self.enh_model.use_concate:
            pred_feats_magnitude = self.enh_model.concate(pred_feats_magnitude, feats_magnitude)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        
        # magnitude -> spectrogram
        pred_feats_spectrogram = pred_feats_magnitude**2
                

        if self.enh_model.use_concate_fbank:
            input_feats, _ = self.logmel(pred_feats_spectrogram, feats_lens)
            input_feats_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            input_feats = torch.cat((input_feats,input_feats_noisy),dim = 2)
        elif self.enh_model.use_dsrn_fbank:
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_noise_spectrogram = feature_noise_pre**2
            pred_fbank_enhanced, _ = self.logmel(pred_feats_spectrogram, feats_lens)
            pred_fbank_noise, _ = self.logmel(pred_noise_spectrogram, feats_lens)
            fbank_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            pred_fbank_enhanced, pred_fbank_noise = self.enh_model.dsrn(enhanced=pred_fbank_enhanced, noise=pred_fbank_noise)
            
            input_feats = pred_fbank_enhanced
        else:
            input_feats, _ = self.logmel(pred_feats_spectrogram, feats_lens)

        
        if self.enh_model.use_gru:
            input_feats_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            input_feats = self.enh_model.forward_gru(input_feats, input_feats_noisy)


        # 2. ASR or ST

        loss_asr, stats, weight = self.s2t_model(
            input_feats, feats_lens, text, text_lengths
        )
        

        if loss_enh is not None and loss_dsrn is not None:
            loss = self.weight_enh*loss_enh + loss_asr + self.weight_refine*loss_dsrn
            #loss = loss_asr+ 15*loss_dsrn            
        elif loss_enh is not None and loss_dsrn is None:
            loss = loss_asr + self.weight_enh*loss_enh
        else:
            loss = loss_asr
            
        
        stats["loss"] = loss.detach() if loss is not None else None
        stats["loss_asr"] = loss_asr.detach() if loss_asr is not None else None
        stats["loss_enh"] = loss_enh.detach() if loss_enh is not None else None
        stats["loss_dsrn"] = loss_dsrn.detach() if loss_dsrn is not None else None

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            ret = self.s2t_model.collect_feats(
                speech,
                speech_lengths,
                text,
                text_lengths,
                **kwargs,
            )
            feats, feats_lengths = ret["feats"], ret["feats_lengths"]
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    # 推理的时候调用
    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # clean: torch.Tensor, clean_lengths: torch.Tensor
        feats, feats_lens = self._compute_stft(speech, speech_lengths)
        feats_spectrogram = feats.real**2 + feats.imag**2
        feats_magnitude = torch.sqrt(feats_spectrogram)
                
        if False:
            #clean, feats_lens = self._compute_stft(clean, clean_lengths)
            #clean_spectrogram = clean.real**2 + clean.imag**2
            #clean_magnitude = torch.sqrt(clean_spectrogram)
            #feats_magnitude[:,:,128:257] = clean_magnitude[:,:,128:257]
            #print("done")
            #feats_spectrogram = feats_magnitude**2
            feats_spectrogram_ns = feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/snr-10/noisy.txt',feats_spectrogram_ns)
        if False:
            feats_spectrogram_ns = feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/snr-10/noisy.txt',feats_spectrogram_ns) 
        
        if False:    
            clean, feats_lens = self._compute_stft(clean, clean_lengths)
            clean_magnitude = clean.real**2 + clean.imag**2
            clean_spectrogram = clean_magnitude**2
            clean_spectrogram_ns = clean_spectrogram.numpy()
            clean_spectrogram_ns= np.reshape(clean_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/clean.txt',clean_spectrogram_ns)
            raise "saving is finished"

        # Enhance
        pred_mask = self.enh_model.forward_enhance(
            feats_magnitude, feats_lens
        )
        pred_feats_magnitude = pred_mask*feats_magnitude
        
        
        #if self.enh_model.save_spec:
        if False:
            pred_feats_spectrogram = pred_feats_magnitude**2
            feats_spectrogram_ns = pred_feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/snr0/before_refine_no_drop.txt',feats_spectrogram_ns)
            raise
            

        # dsrn
        if self.enh_model.use_dsrn and not self.enh_model.use_dsrn_fbank:
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude, pre_noise_magnitude = self.enh_model.dsrn(enhanced=pred_feats_magnitude, noise=feature_noise_pre)
            pred_feats_spectrogram = pred_feats_magnitude**2
        elif self.enh_model.use_dsrn_linear_mask:
            pred_mask, pred_noise_mask = self.enh_model.dsrn(enhanced=pred_mask, noise=1 - pred_mask)
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude = pred_mask * feats_magnitude              #pred_feats_magnitude = pred_mask * pred_feats_magnitude
            pre_noise_magnitude =  pred_noise_mask * feats_magnitude        #pre_noise_magnitude =  pred_noise_mask * feature_noise_pre
            pred_feats_spectrogram = pred_feats_magnitude**2
        elif self.enh_model.use_dsrn_conv2d_mask:
            pred_mask, pred_noise_mask = self.enh_model.dsrn(enhanced=pred_mask, noise=1 - pred_mask)
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude = pred_mask * feats_magnitude              #pred_feats_magnitude = pred_mask * pred_feats_magnitude
            pre_noise_magnitude =  pred_noise_mask * feats_magnitude        #pre_noise_magnitude =  pred_noise_mask * feature_noise_pre
            pred_feats_spectrogram = pred_feats_magnitude**2
        else:
            pred_feats_spectrogram = pred_feats_magnitude**2    

        if self.enh_model.use_concate:
            pred_feats_magnitude = self.enh_model.concate(pred_feats_magnitude, feats_magnitude)  
            pred_feats_spectrogram = pred_feats_magnitude**2
        if False:
            pred_feats_spectrogram = pred_feats_magnitude**2
            feats_spectrogram_ns = pred_feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/after_refine_no_drop.txt',feats_spectrogram_ns)
            raise

        #pred_feats_spectrogram_enh = pred_feats_spectrogram.numpy()
        #pred_feats_spectrogram_enh= np.reshape(pred_feats_spectrogram_enh,[-1,257])
        #np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr1/enh.txt',pred_feats_spectrogram_enh)

        # 计算fbank特征
        if self.enh_model.use_concate_fbank:
            input_feats, _ = self.logmel(pred_feats_spectrogram, feats_lens)
            input_feats_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            input_feats = torch.cat((input_feats,input_feats_noisy),dim = 2)
        elif self.enh_model.use_dsrn_fbank:
            feature_noise_pre = feats_magnitude-pred_feats_magnitude
            pred_noise_spectrogram = feature_noise_pre**2
            pred_fbank_enhanced, _ = self.logmel(pred_feats_spectrogram, feats_lens)
            pred_fbank_noise, _ = self.logmel(pred_noise_spectrogram, feats_lens)
            fbank_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            pred_fbank_enhanced, pred_fbank_noise = self.enh_model.dsrn(enhanced=pred_fbank_enhanced, noise=pred_fbank_noise)

            input_feats = pred_fbank_enhanced
        else:
            input_feats, _ = self.logmel(pred_feats_spectrogram, feats_lens)


        if self.enh_model.use_gru:
            input_feats_noisy, _ = self.logmel(feats_spectrogram, feats_lens)
            input_feats = self.enh_model.forward_gru(input_feats, input_feats_noisy)
        
        encoder_out, encoder_out_lens = self.s2t_model.encode(
            input_feats, feats_lens
        )

        return encoder_out, encoder_out_lens
    # 用transformer的decoder计算负对数自然似然值
    #  这个函数称作分批化负对数似然
    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        return self.s2t_model.nll(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
        )

    batchify_nll = ESPnetASRModel.batchify_nll
    
    # 继承属性
    def inherite_attributes(
        self,
        inherite_enh_attrs: List[str] = [],
        inherite_s2t_attrs: List[str] = [],
    ):
        assert check_argument_types()

        if len(inherite_enh_attrs) > 0:
            for attr in inherite_enh_attrs:
                setattr(self, attr, getattr(self.enh_model, attr, None))
        if len(inherite_s2t_attrs) > 0:
            for attr in inherite_s2t_attrs:
                setattr(self, attr, getattr(self.s2t_model, attr, None))
    
    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
         
        input = input[:, : input_lengths.max()]
        input_stft, feats_lens = self.stft(input, input_lengths)
        
        # "2" refers to the real/imag parts of Complex
        assert input_stft.dim() >= 4, input_stft.shape
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        # 返回复数张量
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
