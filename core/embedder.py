from typing import List, Optional
import os
import numpy as np
from pathlib import Path
from langchain_core.embeddings import Embeddings
from config.settings import settings

# 禁用 Hugging Face Hub 联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安装，请运行: pip install transformers")


class BGEEmbedder(Embeddings):
    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            batch_size: Optional[int] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 transformers: pip install transformers")

        model_name = model_name or settings.embedding_model
        device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size

        root_path = Path(__file__).parent.parent
        model_name = model_name.lstrip("./").lstrip(".\\")
        full_model_path = (root_path / model_name).resolve()
        print(f"【Embedding】模型完整路径: {full_model_path}")

        if not full_model_path.is_dir():
            raise FileNotFoundError(f"模型文件夹不存在：{full_model_path}")

        print("正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(full_model_path),
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            str(full_model_path),
            local_files_only=True,
            trust_remote_code=True
        )

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.eval()
        self.model_name = model_name
        self.dimension = 768  # bge-base-zh-v1.5 的维度是 768

        print(f"✅ 模型加载成功！设备: {self.device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_single_batch(self, texts: List[str]) -> np.ndarray:
        """编码单个批次"""
        import torch

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )

        if self.device == "cuda":
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}

        # 推理
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 池化
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # 归一化
        # p = 2：L2范数归一化
        # dim = 1：沿着向量维度做归一化（对每一条文本向量单独归一化）
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        if self.device == "cuda":
            sentence_embeddings = sentence_embeddings.cpu()

        return sentence_embeddings.numpy()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_single_batch(batch)
            all_embeddings.append(batch_embeddings)

        if all_embeddings:
            result = np.vstack(all_embeddings)
            return result.tolist()
        return []

    def embed_query(self, text: str) -> List[float]:
        embedding = self._encode_single_batch([text])
        return embedding[0].tolist()

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])

        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_single_batch(batch)
            all_embeddings.append(batch_embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    @property
    def model_card(self) -> str:
        return f"BGE-M3 ({self.model_name}) | dim={self.dimension} | device={self.device}"