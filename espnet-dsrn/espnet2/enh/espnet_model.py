"""Enhancement model module."""
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.layers.stft import Stft
from torch_complex.tensor import ComplexTensor
from espnet2.enh.dsrn_two_branch import Dsrn_two_branch
from espnet2.enh.dsrn_linear import Dsrn
from espnet2.enh.dsrn_fusion import Dsrn_fusion
from espnet2.enh.dsrn_conv2d import Dsrn_conv2d
from espnet2.enh.concate import Concate
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
        cascaded: bool = False,
        use_dsrn: bool = False,
        use_dsrn_fbank: bool = False, 
        use_dsrn_fusion: bool = False,
        use_dsrn_conv2d: bool = False,
        use_dsrn_conv2d_mask: bool = False,
        use_dsrn_linear_mask: bool = False,
        use_adaptive_weight: bool = False,
        use_adaptive_weight_mae: bool = True,
        use_ratio_two_branch: bool = False,
        use_crnn: bool = False,
        use_gru: bool = False, 
        use_demucs: bool = False, 
        save_spec: bool = False, 
        skip_cal_loss: bool = False, 
        use_concate: bool = False, 
        use_concate_fbank: bool = False, 
        use_asymmetric: bool = False, 
        skip_refine_loss: bool = False,
        use_dcunet: bool = False,
        cal_dcunet_loss: bool = False,
    ):
        assert check_argument_types()

        super().__init__()
        self.encoder = encoder
        #self.separator = separator
        #self.decoder = decoder
        #separator: AbsSeparator,

        #self.num_spk = separator.num_spk
        self.num_spk = 1
        self.loss_wrappers = loss_wrappers
        # get mask type for TF-domain models
        # (only used when loss_type="mask_*") (deprecated, keep for compatibility)
        self.mask_type = mask_type.upper() if mask_type else None
              
        
        # get loss type for model training (deprecated, keep for compatibility)
        self.loss_type = loss_type
        
        # whether to compute the TF-domain loss
        # while enforcing STFT consistency (deprecated, keep for compatibility)
        self.stft_consistency = stft_consistency

        self.stft = Stft(
            n_fft=512,
            win_length=512,
            hop_length=128,
            center=True,
            window="hann",
            normalized=False,
            onesided=True,
            )

        self.cascaded = cascaded
        self.save_spec = save_spec
        self.use_dsrn = use_dsrn
        self.use_dsrn_fbank = use_dsrn_fbank
        self.use_dsrn_fusion = use_dsrn_fusion
        self.use_dsrn_conv2d = use_dsrn_conv2d
        self.use_dsrn_conv2d_mask = use_dsrn_conv2d_mask
        self.use_dsrn_linear_mask = use_dsrn_linear_mask

        self.use_gru = use_gru
        self.use_crnn = use_crnn
        self.use_demucs = use_demucs
        self.use_ratio_two_branch = use_ratio_two_branch
        self.use_adaptive_weight = use_adaptive_weight
        self.skip_cal_loss = skip_cal_loss
        self.use_adaptive_weight_mae = use_adaptive_weight_mae
        self.use_concate = use_concate
        self.use_concate_fbank = use_concate_fbank
        self.use_asymmetric = use_asymmetric
        self.skip_refine_loss = skip_refine_loss
        self.use_dcunet = use_dcunet
        self.cal_dcunet_loss = cal_dcunet_loss

        

        if self.use_concate:
            self.concate = Concate()
        
        if self.use_dsrn and self.use_ratio_two_branch:
            self.dsrn = Dsrn_two_branch( use_adaptive_weight)
        

        if self.use_dsrn and self.use_dsrn_fusion:
            self.dsrn = Dsrn_fusion( use_adaptive_weight)
        elif self.use_dsrn and self.use_dsrn_conv2d:
            self.dsrn = Dsrn_conv2d(use_adaptive_weight, use_adaptive_weight_mae, use_asymmetric)
        elif self.use_dsrn and self.use_dsrn_fbank:
            self.dsrn = Dsrn( use_adaptive_weight, use_adaptive_weight_mae, use_asymmetric)
        elif self.use_dsrn_conv2d_mask:
            self.dsrn = Dsrn_conv2d(use_adaptive_weight, use_adaptive_weight_mae, use_asymmetric)
        elif self.use_dsrn_linear_mask:
            self.dsrn = Dsrn( use_adaptive_weight, use_adaptive_weight_mae, use_asymmetric)
        elif self.use_dsrn:
            self.dsrn = Dsrn( use_adaptive_weight, use_adaptive_weight_mae, use_asymmetric)
        

        if self.use_gru:
            self.gru0 = torch.nn.GRU(input_size=80, hidden_size=80, num_layers=1, batch_first=True)
            self.gru1 = torch.nn.GRU(input_size=80, hidden_size=80, num_layers=1, batch_first=True)
            self.gru2 = torch.nn.GRU(input_size=80, hidden_size=80, num_layers=1, batch_first=True)
            self.gru3 = torch.nn.GRU(input_size=80, hidden_size=80, num_layers=1, batch_first=True)

    
    # 增强网络的前向传播函数
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor ,
        speech_ref1: torch.Tensor = None ,
        speech_ref1_lengths: torch.Tensor = None ,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            kwargs: "utt_id" is among the input.
        """
        # clean speech signal of each speaker
        #speech_ref = [
            #kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        #]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        # 沿一个新维度对输入张量序列进行连接
        #speech_ref = torch.stack(speech_ref, dim=1)
        
        # 噪声参考，当使用beamforming前端时需s要
        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        #dereverb_speech_ref = None
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int().fill_(speech.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech.shape[0] == speech_ref1.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_ref1.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref1 = speech_ref1[..., : speech_lengths.max()]
        #speech_ref1 = speech_ref1.unbind(dim=1)
        # additional = {}
        # Additional data is required in Deep Attractor Network
        #if isinstance(self.separator, DANSeparator):
            #additional["feature_ref"] = [
                #self.encoder(r, speech_lengths)[0] for r in speech_ref
            #]

        speech = speech[:, : speech_lengths.max()]

        feats, feats_lens = self._compute_stft(speech, speech_lengths)
        feats_spectrogram = feats.real**2 + feats.imag**2
        feats_magnitude = torch.sqrt(feats_spectrogram)


        feats_ref1, feats_ref1_lens = self._compute_stft(speech_ref1, speech_lengths)
        feats_ref1_spectrogram = feats_ref1.real**2 + feats_ref1.imag**2
        feats_ref1_magnitude = torch.sqrt(feats_ref1_spectrogram)


        pred_mask = self.forward_enhance(feats_magnitude, feats_lens)
        pred_feats_magnitude = pred_mask*feats_magnitude
        # branch
        # 9.23修改
        #feats_pre = pred_mask 之前的结果都是基于mapping的
        # loss computation based on mapping 
        #loss_se, stats, weight = self.forward_loss(
            #pred_mask,
            #feats_ref1_spectrogram,
            #feats_spectrogram
        #)
        loss_se, stats, weight = self.forward_loss(
            feats_ref1_magnitude,
            feats_magnitude,
            pred_feats_magnitude,
        )
        stats["loss_se"] = loss_se.detach()

        if self.use_dsrn:
            pre_noise_magnitude = feats_magnitude-pred_feats_magnitude
            pred_feats_magnitude, pre_noise_magnitude = self.dsrn.forward(enhanced=pred_feats_magnitude, noise=pre_noise_magnitude)
            loss_dsrn = None
            loss_dsrn = self.dsrn.forward_loss(pred_feats_magnitude, pre_noise_magnitude, feats_ref1_magnitude, feats_magnitude)
            stats["loss_dsrn"] = loss_dsrn.detach()
            loss = loss_se + loss_dsrn
        else:
            loss = loss_se

        stats["loss"] = loss.detach()

        return loss, stats, weight
    
    # 语音增强，语音增强一般在时频域上增强，增强时只需要输入带噪语音
    # 在鲁棒语音识别的时候调用
    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 混合语音特征
        pred_mask, flens = self.encoder(speech_mix, speech_lengths)


        #if feature_pre is not None:
            #speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        #else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            #speech_pre = None
        #return speech_pre, feature_mix, feature_pre, others
        return pred_mask

    def forward_dcunet(
        self,
        speech_mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 混合语音特征
        # print(speech_mix.shape) torch.Size([5, 101504])
        speech_mix_stft = torch.stft(input=speech_mix, n_fft=self.encoder.n_fft, hop_length=self.encoder.hop_length, normalized=True)
        
        subsampled_input, subsampled_target = self.encoder.subsample2(speech_mix)
        #print(subsampled_input.shape)  torch.Size([5, 50624])
        #print(subsampled_target.shape) torch.Size([5, 50624])
        subsampled_input, subsampled_target = subsampled_input.type(torch.cuda.FloatTensor),subsampled_target.type(torch.cuda.FloatTensor)

        # [B,L] > [B,F,T]
        subsampled_input_stft = torch.stft(input=subsampled_input, n_fft=self.encoder.n_fft, hop_length=self.encoder.hop_length, normalized=True)
        input_stft = torch.stft(input=speech_mix, n_fft=self.encoder.n_fft, hop_length=self.encoder.hop_length, normalized=True)
        
        # print(subsampled_input_stft.shape) torch.Size([5, 512, 198, 2])
        enhanced_subsampled_input = self.encoder(subsampled_input_stft)
        # print(enhanced_subsampled_input.shape) torch.Size([5, 50432])
        #enhanced_subsampled_input = enhanced_subsampled_input[..., : subsampled_input.shape[1]]
        B,L = subsampled_input.shape
        L = subsampled_input.shape[1] - enhanced_subsampled_input.shape[1]
        pad = torch.zeros([B,L]).to('cuda')
        enhanced_subsampled_input = torch.cat((enhanced_subsampled_input, pad), dim=1)
        #print(enhanced_subsampled_input.shape) torch.Size([5, 50624])
        

        with torch.no_grad():
            # input_stft = input_stft
            enhanced_input = self.encoder(input_stft)
            subsampled_enhanced_input, subsampled_enhanced_target = self.encoder.subsample2(enhanced_input)
            subsampled_enhanced_input, subsampled_enhanced_target = subsampled_enhanced_input.type(torch.cuda.FloatTensor), subsampled_enhanced_target.type(torch.cuda.FloatTensor)
            
            enhanced_speech = self.encoder(speech_mix_stft)
        # print(subsampled_target.shape) torch.Size([5, 50624])
        loss = self.encoder.loss_fn(subsampled_input, enhanced_subsampled_input, subsampled_target, subsampled_enhanced_input, subsampled_enhanced_target)
        
        return enhanced_speech, loss

    def forward_loss(
        self,
        feats_ref1: torch.Tensor,
        feats_mix: torch.Tensor,
        feats_pre: torch.Tensor,
        pred_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        loss = 0.0
        stats = dict()
        o = {}
        # feature_mix是带噪的
        # speech_ref是参考的干净的语音
        # loss_wrapper 为 fixed_order 
        # loss_wrapper.criterion 为 FrequencyDomainMSE
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if isinstance(criterion, TimeDomainLoss):
                # for the time domain criterions
                l, s, o = loss_wrapper(feats_ref1, feats_mix)
            elif isinstance(criterion, FrequencyDomainLoss):
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute loss on masks
                    # 用带噪语谱图和干净语谱图计算真实mask标签
                    tf_ref = criterion.create_mask_label(
                        mix_spec=feats_mix,
                        ref_spec=feats_ref1,
                    )
                    tf_pre = pred_mask
                else:
                    # compute on magnitude
                    #tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in speech_ref]
                    tf_ref = feats_ref1
                    tf_pre = feats_pre
                l, s, o = loss_wrapper(tf_ref, tf_pre)
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight 
            stats.update(s)

        stats["loss"] = loss.detach()
        

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = feats_ref1.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    
    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
    
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
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens

    def forward_gru(self, enhanced: torch.Tensor, noisy: torch.Tensor):
        

        B , T , F = enhanced.shape
        output = enhanced
        #device = 'cuda:0'
        device = torch.cuda.current_device()
        h0 = torch.randn(1, B, 80).to(device)
        #h0 = torch.randn(1, B, 80)
        noisy_repre, h1 = self.gru0(noisy, h0)
        enh_repre, h1 = self.gru0(enhanced, h1)

        noisy_repre, h2 = self.gru1(noisy, h1)
        enh_repre, h2 = self.gru1(enhanced, h2)
        
        noisy_repre, h3 = self.gru2(noisy, h2)
        enh_repre, h3 = self.gru2(enhanced, h3)

        noisy_repre, h4 = self.gru3(noisy, h3)
        enh_repre, h4 = self.gru3(enhanced, h4)


        h4_cat = torch.rand(B, T, F).to(device)
        #h4_cat = torch.rand(B, T, F)
        for i in range(T):
            h4_cat[:,i,:] = h4

        output = torch.cat((noisy_repre, h4_cat, enh_repre), 2).to(device)
        #output = torch.cat((noisy_repre, h4_cat, enh_repre), 2)


        #output = torch.cat((noisy_repre, enh_repre), 2).to(device)

        return output 