import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from model import VQAModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = []
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question.append(self.question2idx[word])
            except KeyError:
                question.append(len(self.question2idx))  # 未知語

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.LongTensor(question), torch.LongTensor(answers), int(mode_answer_idx)

        else:
            return image, torch.LongTensor(question)

    def __len__(self):
        return len(self.df)

    def collate_fn(self, data):
        if self.answer:
            image, question, answers, mode_answer_idx = zip(*data)
            question = torch.nn.utils.rnn.pad_sequence(
                question,
                batch_first=True,
                padding_value=len(self.question2idx)+1,
            )
            return (
                torch.stack(image),
                question,
                torch.stack(answers),
                torch.LongTensor(mode_answer_idx)
            )

        else:
            image, question = zip(*data)
            question = torch.nn.utils.rnn.pad_sequence(
                question,
                batch_first=True,
                padding_value=len(self.question2idx)+1,
            )
            return torch.stack(image), question


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルの実装


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    train_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            torch.nn.ModuleList([
                # transforms.RandomAffine(
                #     degrees=(-15, 15), translate=(0.0, 0.3), scale=(0.8, 1.2)
                # ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.3))
            ]), p=0.5
        ),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train")
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=eval_transform, answer=False)
    train_subset, valid_subset = (
        torch.utils.data.random_split(
            dataset=train_dataset,
            lengths=[len(train_dataset) - int(len(train_dataset) * 0.1), int(len(train_dataset) * 0.1)],
            generator=torch.Generator().manual_seed(42)
        )
    )
    train_subset.dataset.transform = train_transform
    valid_subset.dataset.transform = eval_transform
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn,
        pin_memory=True, persistent_workers=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn,
        pin_memory=True, persistent_workers=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn,
        pin_memory=True, persistent_workers=True, num_workers=4
    )

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx)+1,  # 未知語
        n_answer=len(train_dataset.answer2idx)
    ).to(device)
    print(model)

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s], "
              f"train loss: {train_loss:.4f}, "
              f"train acc: {train_acc:.4f}, "
              f"train simple acc: {train_simple_acc:.4f}")
        valid_loss, valid_acc, valid_simple_acc, valid_time = eval(model, valid_loader, optimizer, criterion, device)
        print(f"valid time: {valid_time:.2f} [s], "
              f"valid loss: {valid_loss:.4f}, "
              f"valid acc: {valid_acc:.4f}, "
              f"valid simple acc: {valid_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model, "models/model.pth")
    np.save("submission.npy", submission)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
