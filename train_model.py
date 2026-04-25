import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset safely
df = pd.read_csv("movies.csv", low_memory=False)

print("Columns in dataset:", df.columns)

# If 'overview' exists use it, otherwise use 'description'
if 'overview' in df.columns:
    df['overview'] = df['overview'].fillna('')
    df['tag'] = df['overview']
else:
    raise Exception("No text column found like 'overview'")

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tag'])

# Create indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Save files
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl','wb'))
pickle.dump(indices, open('indices.pkl','wb'))
df.to_pickle('df.pkl')
pickle.dump(tfidf, open('tfidf.pkl','wb'))

print("Total movies:", len(df))
print("Null tag values:", df['tag'].isnull().sum())
print("Sample tag text:", df['tag'].iloc[0])










print("Model training completed successfully ✅")