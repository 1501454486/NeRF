import torch
from src.models.encoding.freq import Encoder as FreqEncoder
from src.config import cfg as global_cfg


def get_encoder(cfg):
    if cfg.type == "frequency":
        encoder_kwargs = {
            "include_input": True,
            "input_dims": cfg.input_dim,
            "max_freq_log2": cfg.freq - 1,
            "num_freqs": cfg.freq,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        encoder_obj = FreqEncoder(**encoder_kwargs)
        encoder = lambda x, eo=encoder_obj: eo.embed(x)
        return encoder, encoder_obj.out_dim
    elif cfg.type == "cuda_hashgrid":
        from src.models.encoding.hashencoder.hashgrid import HashEncoder

        encoder = HashEncoder(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_hashgrid_latent":
        from src.models.encoding.hashencoder.hashgrid import HashLatent

        encoder = HashLatent(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_hashgrid_4d":
        from src.models.encoding.hashencoder.hashgrid import HashEncoder4d

        encoder = HashEncoder4d(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_hashgrid_coef":
        from src.models.encoding.hashencoder.hashgrid import HashEncoderCoef

        encoder = HashEncoderCoef(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_triplane":
        from src.models.encoding.hashencoder.hashgrid import TriPlane

        encoder = TriPlane(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_motion2d":
        from src.models.encoding.hashencoder.hashgrid import Motion2d

        encoder = Motion2d(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "cuda_dnerf_ngp_tensorf":
        from src.models.encoding.hashencoder.hashgrid import DNeRFNGP

        encoder = DNeRFNGP(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "dnerf_ngp_tensorf":
        from src.models.encoding.hashgrid import DNeRFNGP

        encoder = DNeRFNGP(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "dnerf_ngp_mlp":
        from src.models.encoding.hashgrid import DNeRFNGP_MLP

        encoder = DNeRFNGP_MLP(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "dnerf_mlp_tensorf":
        from src.models.encoding.hashgrid import DNeRFTensoRF

        encoder = DNeRFTensoRF(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "triplane":
        from src.models.encoding.triplane import TriPlane

        encoder = TriPlane(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "dnerf":
        from src.models.encoding.dnerf import DNeRF

        encoder = DNeRF(**cfg)
        return encoder, encoder.out_dim
    elif cfg.type == "sphere_harmonics":
        pass
    elif cfg.type == "grid_hash":
        cfg.bbox = global_cfg.bbox
        encoder = GridHashEncoder(cfg)
        return encoder, encoder.out_dim
    else:
        raise NotImplementedError
