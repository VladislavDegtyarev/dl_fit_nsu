import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoTokenizer, AutoModel

from model import SentimentClassifier

def main():
    wandb.login()
    
    seed_everything(42)
    config = {
        'pretrained_model': 'DeepPavlov/rubert-base-cased',
        'num_classes': 5,
        'dropout': 0.2,
        'max_len': 32,
        'batch_size': 64,
        'lr': 2e-5,
        'epochs': 3,
        'num_workers': 8,
        'csv_path': 'dataset/rusentitweet_full.csv',
        'val_step_frequency': 75,
        'freeze_pretrained': False,
    }

    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
    config['tokenizer'] = tokenizer

    wandb.init(project='sentiment_analysis', config=config)
    wandb_logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath='checkpoints',
        filename='sentiment-{epoch:02d}-{val_f1:.2f}',
        save_top_k=1,
        mode='max',
    )

    model = SentimentClassifier(config)
    trainer = Trainer(
        val_check_interval=90,
        check_val_every_n_epoch=None,
        logger=wandb_logger,
        max_epochs=config['epochs'],
        gpus=1,
        deterministic=True,
        precision=16,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model)

    trainer.test(dataloaders=model.test_dataloader())
    
if __name__ == '__main__':
    main()