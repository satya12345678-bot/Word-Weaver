import os
import sys
import re
import subprocess
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import string
import nltk.tokenize

def install_libraries():
    libraries = ['openpyxl', 'pandas', 'requests', 'beautifulsoup4', 'nltk']
    for library in libraries:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
            print(f"Successfully installed {library}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {library}")

def read_excel_file(input_file):
    try:
        return pd.read_excel(input_file)
    except ImportError:
        try:
            import xlrd
            return pd.read_excel(input_file, engine='xlrd')
        except ImportError:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(input_file)
                sheet = wb.active
                data = []
                headers = [cell.value for cell in sheet[1]]
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    data.append(dict(zip(headers, row)))
                return pd.DataFrame(data)
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                raise

def extract_article_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')

        title_element = soup.find('h1', class_='entry-title')
        article_title = title_element.get_text(strip=True) if title_element else 'No Title Found'
        content_element = soup.find('div', class_='td-post-content tagdiv-type')

        if content_element:
            for script_or_style in content_element(['script', 'style']):
                script_or_style.decompose()
                
            article_text = '\n\n'.join([p.get_text(strip=True) for p in content_element.find_all('p') if p.get_text(strip=True)])
        else:
            article_text = 'No Content Found'
        
        return article_title, article_text
    
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None, None

class SentimentAnalyzer:
    def __init__(self, stopwords_dir, master_dict_dir):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stopwords = self._load_stopwords(stopwords_dir)
        self.positive_words = self._load_dictionary(os.path.join(master_dict_dir, 'positive-words.txt'))
        self.negative_words = self._load_dictionary(os.path.join(master_dict_dir, 'negative-words.txt'))
        self.personal_pronouns_regex = r'\b(I|we|my|ours|us)(?!\s*[A-Z])\b'
    
    def _load_stopwords(self, stopwords_dir):
        stopwords_set = set()
        for filename in os.listdir(stopwords_dir):
            filepath = os.path.join(stopwords_dir, filename)
            with open(filepath, 'r', encoding='latin-1') as f:
                stopwords_set.update(line.strip().lower() for line in f if line.strip())
        return stopwords_set
    
    def _load_dictionary(self, filepath):
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: Dictionary file {filepath} not found.")
            return set()
    
    def _clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        try:
            words = nltk.word_tokenize(text)
        except LookupError:
            words = text.split()
        cleaned_words = [word for word in words if word not in self.stopwords and word.isalnum()]
        
        return cleaned_words
    
    def _count_syllables(self, word):
        if word.endswith(('es', 'ed')):
            word = word[:-2]
        vowels = 'aeiou'
        count = 0
        last_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not last_was_vowel:
                count += 1
            last_was_vowel = is_vowel
        return max(1, count)
    
    def analyze(self, text):
        cleaned_words = self._clean_text(text)
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            sentences = text.split('.')

        # Defining output vars:
        positive_score = sum(1 for word in cleaned_words if word in self.positive_words)
        negative_score = sum(1 for word in cleaned_words if word in self.negative_words)
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)
        avg_sentence_length = len(cleaned_words) / max(1, len(sentences))
        complex_words = [word for word in cleaned_words if self._count_syllables(word) > 2]
        complex_word_count = len(complex_words)
        percentage_complex_words = complex_word_count / max(1, len(cleaned_words))
        fog_index = 0.4 * (avg_sentence_length + (percentage_complex_words * 100))
        avg_words_per_sentence = len(cleaned_words) / max(1, len(sentences))
        word_count = len(cleaned_words)
        syllables_per_word = sum(self._count_syllables(word) for word in cleaned_words) / max(1, len(cleaned_words))
        personal_pronouns = len(re.findall(self.personal_pronouns_regex, text, re.IGNORECASE))
        avg_word_length = sum(len(word) for word in cleaned_words) / max(1, len(cleaned_words))
        
        return {
            'POSITIVE_SCORE': positive_score,
            'NEGATIVE_SCORE': negative_score,
            'POLARITY_SCORE': polarity_score,
            'SUBJECTIVITY_SCORE': subjectivity_score,
            'AVG_SENTENCE_LENGTH': avg_sentence_length,
            'PERCENTAGE_COMPLEX_WORDS': percentage_complex_words,
            'FOG_INDEX': fog_index,
            'AVG_WORDS_PER_SENTENCE': avg_words_per_sentence,
            'COMPLEX_WORD_COUNT': complex_word_count,
            'WORD_COUNT': word_count,
            'SYLLABLES_PER_WORD': syllables_per_word,
            'PERSONAL_PRONOUNS': personal_pronouns,
            'AVG_WORD_LENGTH': avg_word_length
        }

def extract_articles_from_excel(input_file, output_dir='extracted_articles'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        df = read_excel_file(input_file)
    except Exception as e:
        print(f"Could not read Excel file: {e}")
        install_libraries()
        df = read_excel_file(input_file)

    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        
        print(f"Processing URL_ID: {url_id}")
    
        article_title, article_text = extract_article_text(url)
        
        if article_text:
            output_file = os.path.join(output_dir, f"{url_id}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article_title}\n\n")
                f.write(article_text)
            
            print(f"Saved article for URL_ID: {url_id}")
        else:
            print(f"Failed to extract article for URL_ID: {url_id}")
    
    return output_dir

def process_all_articles(input_excel_path, articles_dir, stopwords_dir, master_dict_dir, output_csv_path):
    input_df = pd.read_excel(input_excel_path)
    analyzer = SentimentAnalyzer(stopwords_dir, master_dict_dir)
    article_files = [f for f in os.listdir(articles_dir) if f.endswith('.txt')]
    if not article_files:
        print("No article files found in the directory.")
        return
    
    results = []
    default_result = {
        'URL_ID': '',
        'URL': '',
        'POSITIVE_SCORE': 0,
        'NEGATIVE_SCORE': 0,
        'POLARITY_SCORE': 0,
        'SUBJECTIVITY_SCORE': 0,
        'AVG_SENTENCE_LENGTH': 0,
        'PERCENTAGE_COMPLEX_WORDS': 0,
        'FOG_INDEX': 0,
        'AVG_WORDS_PER_SENTENCE': 0,
        'COMPLEX_WORD_COUNT': 0,
        'WORD_COUNT': 0,
        'SYLLABLES_PER_WORD': 0,
        'PERSONAL_PRONOUNS': 0,
        'AVG_WORD_LENGTH': 0
    }
    for article_file in article_files:
        url_id = os.path.splitext(article_file)[0]
        
        try:
            row = input_df[input_df['URL_ID'].astype(str) == str(url_id)].iloc[0]
            url = row['URL']
            article_path = os.path.join(articles_dir, article_file)
            with open(article_path, 'r', encoding='utf-8') as f:
                f.readline()
                text = f.read().strip()
            analysis_result = analyzer.analyze(text)
            result = {**default_result.copy(), 'URL_ID': url_id, 'URL': url, **analysis_result}
            results.append(result)
            
            print(f"Processed article (URL_ID: {url_id})")
        
        except Exception as e:
            print(f"Error processing article {url_id}: {e}")
            result = default_result.copy()
            result['URL_ID'] = url_id
            results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nProcessed {len(results)} articles. Results saved to {output_csv_path}")

def main(input_excel_path, stopwords_dir, master_dict_dir, output_csv_path, articles_dir='extracted_articles'):
    extracted_articles_dir = extract_articles_from_excel(input_excel_path, articles_dir)
    process_all_articles(
        input_excel_path, 
        extracted_articles_dir, 
        stopwords_dir, 
        master_dict_dir, 
        output_csv_path
    )

if __name__ == "__main__":
    # Predefined paths (replace with your actual paths)
    input_excel_path = r'Give input'
    stopwords_dir = r'Give input'
    master_dict_dir = r'Give input'
    output_csv_path = r'Give input'
    
    try:
        main(
            input_excel_path, 
            stopwords_dir, 
            master_dict_dir, 
            output_csv_path
        )
    except ImportError:
        install_libraries()
        main(
            input_excel_path, 
            stopwords_dir, 
            master_dict_dir, 
            output_csv_path
        )