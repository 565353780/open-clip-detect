import os
import torch
import open_clip
import numpy as np
from PIL import Image
from typing import Union
from open_clip.factory import PreprocessCfg, image_transform_v2

from open_clip_detect.Config.clip_vision_cfg import CLIPVisionCfg
from open_clip_detect.Config.clip_text_cfg import CLIPTextCfg
from open_clip_detect.Model.clip import CLIP


class Detector(object):
    def __init__(self, model_file_path: Union[str, None]=None, device: str = 'cpu', auto_download_model: bool = False) -> None:
        self.device = device
        self.dtype = torch.float32

        if auto_download_model:
            if model_file_path is not None:
                if os.path.exists(model_file_path):
                    pretrained = model_file_path
            else:
                pretrained = 'dfn5b'

            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained=pretrained)
        else:
            self.model = CLIP(
                embed_dim=1024,
                vision_cfg=CLIPVisionCfg(
                    image_size=378,
                    layers=32,
                    width=1280,
                    head_width=80,
                    patch_size=14,
                ),
                text_cfg=CLIPTextCfg(
                    context_length=77,
                    vocab_size=49408,
                    width=1024,
                    heads=16,
                    layers=24,
                ),
                quick_gelu=True,
            )
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu')

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict, strict=True)

        print('[INFO][Detector::loadModel]')
        print('\t load model success!')
        print('\t model_file_path:', model_file_path)
        return True

    @torch.no_grad()
    def detectImage(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())

        image = image.convert('RGB')

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        clip_image_feature = self.model.encode_image(image_tensor)

        return clip_image_feature

    @torch.no_grad()
    def detectImageFile(self, image_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image = Image.open(image_file_path)

        dino_feature = self.detectImage(image)

        return dino_feature
