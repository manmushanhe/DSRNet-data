        if False:
            clean, feats_lens = self._compute_stft(clean, clean_lengths)
            clean_spectrogram = clean.real**2 + clean.imag**2
            clean_magnitude = torch.sqrt(clean_spectrogram)
            feats_magnitude[:,:,129:256] = clean_magnitude[:,:,129:256]
            print("done")
            feats_spectrogram = feats_magnitude**2
            feats_spectrogram_ns = feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/before_enh_add_clean129-256_snr0.txt',feats_spectrogram_ns)
        if False:
            feats_spectrogram_ns = feats_spectrogram.numpy()
            feats_spectrogram_ns= np.reshape(feats_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/noisy.txt',feats_spectrogram_ns) 
            
            clean, feats_lens = self._compute_stft(clean, clean_lengths)
            clean_magnitude = clean.real**2 + clean.imag**2
            clean_spectrogram = clean_magnitude**2
            clean_spectrogram_ns = clean_spectrogram.numpy()
            clean_spectrogram_ns= np.reshape(clean_spectrogram_ns,[-1,257])
            np.savetxt('/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/spectrogram/clean.txt',clean_spectrogram_ns)
            raise "saving is finished"