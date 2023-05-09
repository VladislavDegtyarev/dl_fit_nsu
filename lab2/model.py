import os
import re
import string
from typing import Optional, Sequence
from string import punctuation
import warnings

import emoji
import nltk
import torch
import wandb

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from transformers import AutoTokenizer, AutoModel

# prepare nltk
nltk.download('punkt')
nltk.download('stopwords')

torch.set_float32_matmul_precision('high')

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
    
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user @ references and '#' from text
    text = re.sub(r'\@\w+|\#', '', text)

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) # no emoji
                
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text


class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        self.data['raw_text'] = self.data['text'].copy()
        self.data['text'] = self.data['text'].apply(clean_text)
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text = row['text']
        raw_text = row['raw_text']
        label = row['label']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,
            'raw_text': raw_text,
        }
    
    
def create_train_val_test_datasets(csv_file, tokenizer, max_len, val_ratio=0.1, test_ratio=0.1):
    data = pd.read_csv(csv_file)
    train_data, temp_data = train_test_split(data, test_size=val_ratio+test_ratio, random_state=42, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(val_ratio+test_ratio), random_state=42, stratify=temp_data['label'])
    train_dataset = SentimentDataset(train_data, tokenizer, max_len)
    val_dataset = SentimentDataset(val_data, tokenizer, max_len)
    test_dataset = SentimentDataset(test_data, tokenizer, max_len)
    return train_dataset, val_dataset, test_dataset


class SentimentClassifier(LightningModule):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()
        self.save_hyperparameters(config)
        self.model = AutoModel.from_pretrained(config['pretrained_model'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(self.model.config.hidden_size, config['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_f1 = F1Score(num_classes=config['num_classes'], task='multiclass')
        self.val_f1 = F1Score(num_classes=config['num_classes'], task='multiclass')
        self.test_f1 = F1Score(num_classes=config['num_classes'], task='multiclass')
        self.val_step_frequency = config['val_step_frequency']
        self.freeze_pretrained = config['freeze_pretrained']
        
        if self.freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def train_dataloader(self):
        train_dataset, _, _ = create_train_val_test_datasets(self.hparams.csv_path, self.hparams.tokenizer, self.hparams.max_len)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        _, val_dataset, _ = create_train_val_test_datasets(self.hparams.csv_path, self.hparams.tokenizer, self.hparams.max_len)
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        _, _, test_dataset = create_train_val_test_datasets(self.hparams.csv_path, self.hparams.tokenizer, self.hparams.max_len)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        train_f1 = self.train_f1(preds, labels)
        self.log('train_loss', loss)
        self.log('train_f1', train_f1)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        val_f1 = self.val_f1(preds, labels)
        self.log('val_loss', loss)
        self.log('val_f1', val_f1)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        test_f1 = self.test_f1(preds, labels)
        self.log('test_loss', loss)
        self.log('test_f1', test_f1)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs)
        return [optimizer], [scheduler]