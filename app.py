import os
import json
import nltk
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import datetime
import base64
from io import BytesIO

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-for-development")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment with NLTK's VADER
def analyze_sentiment_vader(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'score': compound_score,
        'positive': sentiment_scores['pos'],
        'negative': sentiment_scores['neg'],
        'neutral': sentiment_scores['neu']
    }

# Function to analyze sentiment with TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

# Function to get consensus sentiment from multiple models
def get_consensus_sentiment(text):
    vader_result = analyze_sentiment_vader(text)
    textblob_result = analyze_sentiment_textblob(text)
    
    # Combine results from both sentiment analyzers
    result = {
        'vader': vader_result,
        'textblob': textblob_result
    }
    
    # Determine consensus sentiment
    if vader_result['sentiment'] == textblob_result['sentiment']:
        consensus = vader_result['sentiment']
    elif vader_result['sentiment'] == 'neutral' or textblob_result['sentiment'] == 'neutral':
        # If one is neutral, go with the non-neutral one
        consensus = textblob_result['sentiment'] if vader_result['sentiment'] == 'neutral' else vader_result['sentiment']
    else:
        # If they completely disagree, go with the one with the stronger score
        if abs(vader_result['score']) > abs(textblob_result['polarity']):
            consensus = vader_result['sentiment']
        else:
            consensus = textblob_result['sentiment']
    
    result['consensus'] = consensus
    return result

# Function to save analysis to CSV
def save_analysis(text, results):
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Prepare data for saving
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        'timestamp': timestamp,
        'text': text,
        'consensus': results['consensus'],
        'vader_sentiment': results['vader']['sentiment'],
        'vader_score': results['vader']['score'],
        'textblob_sentiment': results['textblob']['sentiment'],
        'textblob_polarity': results['textblob']['polarity'],
        'textblob_subjectivity': results['textblob']['subjectivity']
    }
    
    # Create or append to CSV file
    filename = 'data/sentiment_analysis.csv'
    file_exists = os.path.isfile(filename)
    
    df = pd.DataFrame([data])
    
    if file_exists:
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    
    return data

# Function to generate sentiment distribution chart
def generate_sentiment_chart():
    if not os.path.exists('data/sentiment_analysis.csv'):
        return None
    
    # Read the data
    try:
        df = pd.read_csv('data/sentiment_analysis.csv')
        if len(df) == 0:
            return None
            
        # Count sentiment distribution
        sentiment_counts = df['consensus'].value_counts()
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        # Ensure all sentiments have a count
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
        
        labels = sentiment_counts.index
        plt_colors = [colors[s] for s in labels]
        
        plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', 
                startangle=90, colors=plt_colors, shadow=True)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Sentiment Distribution')
        
        # Save chart to a string buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64 string
        chart_img = base64.b64encode(buffer.read()).decode('utf-8')
        return chart_img
        
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None

# Routes
@app.route('/')
def index():
    # Generate sentiment chart if there's data
    chart_img = generate_sentiment_chart()
    
    # Get recent analyses if they exist
    recent_analyses = []
    if os.path.exists('data/sentiment_analysis.csv'):
        try:
            df = pd.read_csv('data/sentiment_analysis.csv')
            # Get last 5 analyses
            recent_analyses = df.tail(5).to_dict('records')
            # Reverse to show newest first
            recent_analyses.reverse()
        except Exception as e:
            print(f"Error loading recent analyses: {str(e)}")
    
    return render_template('index.html', chart_img=chart_img, recent_analyses=recent_analyses)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided for analysis'})
    
    # Analyze sentiment
    results = get_consensus_sentiment(text)
    
    # Save the analysis
    save_analysis(text, results)
    
    return jsonify(results)

@app.route('/history')
def history():
    if not os.path.exists('data/sentiment_analysis.csv'):
        return render_template('history.html', analyses=[])
    
    try:
        df = pd.read_csv('data/sentiment_analysis.csv')
        analyses = df.to_dict('records')
        
        # Get sentiment distribution for the chart
        sentiment_counts = df['consensus'].value_counts().to_dict()
        
        # Ensure all sentiments have a count
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
                
        return render_template('history.html', analyses=analyses, sentiment_counts=sentiment_counts)
    
    except Exception as e:
        return render_template('history.html', analyses=[], error=str(e))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post is in the request
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
            
        if file and file.filename.endswith('.csv'):
            try:
                # Read the CSV file
                df = pd.read_csv(file)
                
                # Ensure the CSV has a 'text' column
                if 'text' not in df.columns:
                    return render_template('upload.html', error='CSV file must have a "text" column')
                
                # Process each text entry and analyze sentiment
                results = []
                for index, row in df.iterrows():
                    text = row['text']
                    sentiment_results = get_consensus_sentiment(text)
                    
                    # Save each analysis
                    save_analysis(text, sentiment_results)
                    
                    results.append({
                        'text': text,
                        'sentiment': sentiment_results['consensus']
                    })
                
                return render_template('upload.html', success=f'Successfully analyzed {len(results)} entries', results=results)
                    
            except Exception as e:
                return render_template('upload.html', error=f'Error processing file: {str(e)}')
        else:
            return render_template('upload.html', error='File must be a CSV file')
            
    return render_template('upload.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided for analysis',
            'status': 'error'
        }), 400
    
    text = data['text']
    
    # Analyze sentiment
    results = get_consensus_sentiment(text)
    
    # Save the analysis
    save_analysis(text, results)
    
    # Return results
    return jsonify({
        'status': 'success',
        'text': text,
        'results': results
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    if not os.path.exists('data/sentiment_analysis.csv'):
        return jsonify({
            'status': 'success',
            'data': []
        })
    
    try:
        df = pd.read_csv('data/sentiment_analysis.csv')
        analyses = df.to_dict('records')
        
        return jsonify({
            'status': 'success',
            'data': analyses
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)