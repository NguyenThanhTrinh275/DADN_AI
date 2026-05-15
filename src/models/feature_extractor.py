"""
Module trích xuất đặc trưng từ ảnh sử dụng pre-trained CNN
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from tqdm import tqdm

from src.config import config
from src.utils.data_loader import bytes_to_image


class FeatureExtractor:
    """
    Trích xuất đặc trưng từ ảnh sử dụng pre-trained CNN
    
    EfficientNet được ưu tiên vì:
    - Hiệu quả về tính toán (compound scaling)
    - Accuracy cao trên ImageNet
    - EfficientNet-V2-L là phiên bản mạnh nhất trong torchvision
    
    Supported models:
    - efficientnet_v2_l (1280 dims) - Mạnh nhất
    - resnet50 (2048 dims)
    - dinov2_vits14 (384 dims)
    """
    
    SUPPORTED_MODELS = {
        'efficientnet_v2_l': {'input_size': 480, 'feature_dim': 1280},
        'dinov2_vits14':     {'input_size': 518, 'feature_dim': 384},
        'dinov2_vitb14':     {'input_size': 518, 'feature_dim': 768},
        'dinov2_vitl14':     {'input_size': 518, 'feature_dim': 1024},
    }

    @staticmethod
    def _resolve_device(device):
        """Chuẩn hóa device và fallback an toàn khi CUDA không sẵn sàng"""
        requested_device = (device or 'auto').lower()

        if requested_device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'

        if requested_device in ('gpu', 'cuda'):
            if torch.cuda.is_available():
                return 'cuda'
            print("CUDA không khả dụng, tự động chuyển sang CPU.")
            return 'cpu'

        if requested_device == 'cpu':
            return 'cpu'

        raise ValueError("device phải là 'auto', 'cpu', 'gpu' hoặc 'cuda'")
    
    def __init__(self, model_name=None, device=None):
        """
        Khởi tạo Feature Extractor
        
        Args:
            model_name: Tên model (mặc định từ config)
            device: 'auto', 'gpu'/'cuda' hoặc 'cpu' (mặc định: config)
        """
        self.model_name = model_name or config.MODEL_NAME
        self.device = self._resolve_device(device or config.DEVICE)
        
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{self.model_name}' không được hỗ trợ. "
                           f"Các model hỗ trợ: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.input_size = self.SUPPORTED_MODELS[self.model_name]['input_size']
        self.feature_dim = self.SUPPORTED_MODELS[self.model_name]['feature_dim']
        
        self.model = self._load_model()
        self.transform = self._create_transform()
    
    def _load_model(self):
        """Tải pre-trained model và loại bỏ lớp classification"""
        print(f"Loading {self.model_name} model...")
        
        if self.model_name == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
            model.classifier = nn.Identity()
            
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model = nn.Sequential(*list(model.children())[:-1])

        elif self.model_name in ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'):
            try:
                model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Không tải được DINOv2 từ torch.hub. "
                    "Kiểm tra kết nối mạng và thử lại."
                ) from exc
        
        model = model.to(self.device)
        model.eval()

        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        print(f"Model loaded on {self.device}. Feature dimension: {self.feature_dim}")
        return model
    
    def _create_transform(self):
        """Transform mặc định (single-pass)"""
        return self._build_transform(self.input_size, flip=False)

    def _build_transform(self, size, flip=False):
        """Build transform với size và flip tùy chọn (dùng cho TTA)"""
        ops = [
            transforms.Resize(size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
        ]
        if flip:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
        return transforms.Compose(ops)
    
    def extract_features(self, images, batch_size=None):
        """
        Trích xuất features từ danh sách ảnh
        
        Args:
            images: List các bytes ảnh
            batch_size: Kích thước batch (mặc định từ config)
        
        Returns:
            numpy array shape (n_samples, feature_dim)
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        features = []
        
        use_amp = self.device == 'cuda'

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
                batch_images = images[i:i+batch_size]

                batch_tensors = torch.stack([
                    self.transform(bytes_to_image(img)) for img in batch_images
                ]).to(self.device)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = self.model(batch_tensors)
                    out = out.float()
                else:
                    out = self.model(batch_tensors)

                batch_features = out.cpu().numpy().reshape(len(batch_images), -1)
                features.append(batch_features)
        
        features = np.vstack(features)
        features = normalize(features, norm='l2')
        return features
    
    def extract_features_tta(self, images, scales=None, flips=(False, True), batch_size=None):
        """
        Trích xuất features với Test-Time Augmentation (TTA).

        Lấy trung bình features qua nhiều biến thể (multi-scale + hflip),
        sau đó L2-normalize. Tăng độ ổn định và accuracy clustering.

        Args:
            images: List bytes ảnh
            scales: Tuple kích thước input (mặc định = (input_size,))
            flips: Tuple (False, True) để bật/tắt horizontal flip
            batch_size: Batch size khi extract

        Returns:
            numpy array (n_samples, feature_dim) đã L2-normalize
        """
        if scales is None:
            scales = (self.input_size,)

        original_transform = self.transform
        all_feats = []
        try:
            for s in scales:
                for flip in flips:
                    print(f"  TTA pass: scale={s}, flip={flip}")
                    self.transform = self._build_transform(s, flip=flip)
                    feats = self.extract_features(images, batch_size=batch_size)
                    all_feats.append(feats)
        finally:
            self.transform = original_transform

        mean_feats = np.mean(np.stack(all_feats, axis=0), axis=0)
        return normalize(mean_feats, norm='l2')

    def __repr__(self):
        return f"FeatureExtractor(model={self.model_name}, device={self.device}, dim={self.feature_dim})"
