# MBTI
Text analysis on posts to predict the personality types of people.

# Results

Test set accuracy:

RNN: 

Introversion (I) vs Extroversion (E) : 77%

Intuition    (N) vs Sensing      (S) : 86%

Thinking     (T) vs Feeling      (F) : 58%

Judging      (J) vs Perceiving   (P) : 60%

1D CNN: 

Introversion (I) vs Extroversion (E) : 77%

Intuition    (N) vs Sensing      (S) : 86%

Thinking     (T) vs Feeling      (F) : 46%

Judging      (J) vs Perceiving   (P) : 60%

# Visualization

1. Count frequencies of each type.

2. Considered average words per post and variance of words.

3. Created swarm plots, hex plots, and (just for fun) word clouds.

# Preprocess

1. Standard normalization (replaced numbers and urls, lowercased, etc)

2. Split into train, validate, test sets.

3. Mapped letters to integers, and then integers to vectors using GloVe.

4. Split the target into 4 classes.

# Problems

Validation accuracy stops changing very quickly, unless I unfreeze the embedding layer, which will result in overfitting.

I could not get tensorboard histograms and embeddings to work on Keras.

I could not get multi-input to work, so I only used the text data.



# Things to Work On

Try out quick and dirty models from sklearn, and perform ensemble learning.

# Things To Take Away

This is just for fun, but I guess it can be difficult to distinguish between T and F and between J and P.
