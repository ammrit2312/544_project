from evaluate_gec_ft import evaluate
if __name__ == '__main__':
    print('=== English (GECModel / gec_model_encoder_ft) ===')
    evaluate('data/processed/test.csv', sample=200, lang='en')
    print('\n=== Spanish (GECModel / gec_model_encoder_ft) ===')
    evaluate('data/processed/test.csv', sample=200, lang='es')
