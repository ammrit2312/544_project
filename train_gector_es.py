"""
Spanish-only GECTOR: fine-tuned on train.csv rows where lang == 'es'.

Encoder: xlm-roberta-base (multilingual checkpoint; Spanish-only *data*).
Outputs separate checkpoint and label vocab from the multilingual run.
"""

from train_gector import main

ENCODER_NAME = 'xlm-roberta-base'
CHECKPOINT = 'checkpoints/xlm-roberta-base_gector_es.pt'
LABEL_VOCAB = 'checkpoints/xlm-roberta-base_gector_es_labels.pkl'


if __name__ == '__main__':
    main(
        checkpoint=CHECKPOINT,
        label_vocab=LABEL_VOCAB,
        encoder_name=ENCODER_NAME,
        lang='es',
    )
