import numpy as np
import torch
import torch.nn as nn


class VIT(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        prod_patch = np.prod(patch_size)
        prod_image = np.prod(image_size)
        self.n_pat = prod_image // prod_patch
        self.pat_dim = 3 * prod_patch

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = torch.nn.Parameter(torch.randn(1, self.n_pat+1, emb_dim))

        self.proj = nn.Linear(self.pat_dim, emb_dim)
        self.tfs = nn.Sequential(
            *[nn.TransformerEncoderLayer(
                emb_dim,
                8,
                dim_feedforward=2048,
                activation="gelu",
                dropout=0.25,
                batch_first=True
            ) for _ in range(3)]
        )

    def forward(self, image):
        x = image.view(-1, self.n_pat, self.pat_dim)
        x = self.proj(x)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = x + self.pos_emb
        x = self.tfs(x)
        return x[:, 0, :].unsqueeze(dim=1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size+1, emb_dim, padding_idx=vocab_size)
        self.tf1 = nn.TransformerEncoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )
        self.tf2 = nn.TransformerEncoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )
        self.tf3 = nn.TransformerEncoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )

    def forward(self, text):
        x = self.emb(text)

        pos_enc = torch.matmul(
            torch.arange(x.shape[1], dtype=torch.float).view(-1, 1),
            (
                10000 **
                (-(torch.arange(x.shape[2]//2)
                   .repeat_interleave(2))/x.shape[2])
            ).view(1, -1)
        )
        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
        pos_enc = pos_enc.view(1, *pos_enc.shape)
        pad_mask = (text != self.vocab_size).to(dtype=torch.float)
        src_mask = torch.triu(torch.ones(text.shape[1], text.shape[1]) * float('-inf'), diagonal=1)
        pos_enc = pos_enc.to(device=text.device)
        pad_mask = pad_mask.to(device=text.device)
        src_mask = src_mask.to(device=text.device)

        x = x + pos_enc
        x = self.tf1(
            x,
            src_mask=src_mask,
            src_key_padding_mask=pad_mask
        )
        x = self.tf2(
            x,
            src_mask=src_mask,
            src_key_padding_mask=pad_mask
        )
        x = self.tf3(
            x,
            src_mask=src_mask,
            src_key_padding_mask=pad_mask
        )
        return x


class VQAModel(nn.Module):
    def __init__(self, vocab_size, n_answer):
        super().__init__()

        emb_dim = 512

        self.vit = VIT((256, 256), (16, 16), emb_dim)
        self.txt = TextEncoder(vocab_size, emb_dim)

        self.tf1 = nn.TransformerDecoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )
        self.tf2 = nn.TransformerDecoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )
        self.tf3 = nn.TransformerDecoderLayer(
            emb_dim,
            8,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.25,
            batch_first=True
        )
        self.ln = nn.Linear(emb_dim, n_answer)

    def forward(self, image, question):
        image_feature = self.vit(image)  # 画像の特徴量
        question_feature = self.txt(question)  # テキストの特徴量

        x = self.tf1(image_feature, question_feature)
        x = self.tf2(x, question_feature)
        x = self.tf3(x, question_feature)
        x = x.squeeze(dim=1)
        x = self.ln(x)

        return x