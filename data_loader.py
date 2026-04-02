from datasets import load_dataset
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split

def download_english():
    print("Downloading English dataset...")
    dataset = load_dataset("juancavallotti/bea-19-fine-tune")

    df = pd.DataFrame(dataset['train'])

    df = df[['modified', 'sentence', 'transformation']].rename(columns = {
        'modified': 'incorrect',
        'sentence': 'correct',
        'transformation': 'transformation'
    })
    df['lang'] = 'en'
    
    return df

def download_spanish():
    print("Downloading Spanish dataset...")
    dataset = load_dataset("juancavallotti/multilingual-gec")
    
    df = pd.DataFrame(dataset['train'])
    print("Available languages:", df['lang'].unique())
    print("Lang counts:\n", df['lang'].value_counts())

    df = df[df['lang'] == 'es'][['modified', 'sentence', 'transformation']].rename(columns = {
        'modified': 'incorrect',
        'sentence': 'correct',
        'transformation': 'transformation'
    })
    df['lang'] = 'es'
    
    return df
    

def download_dataset():
    df_en = download_english()
    df_es = download_spanish()
    
    df = pd.concat([df_en, df_es], ignore_index=True)
    
    df = df.dropna()
    df = df[df['incorrect'].str.strip() != '']
    df = df[df['correct'].str.strip() != '']

    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print("Dataset downloaded and processed successfully.")
    
    
    print(df[['incorrect','correct']].head(5).to_string())
    
if __name__ == "__main__":
    download_dataset()
