# Stockify

# Problem
Investing in Stocks is great way to grow money
Picking the right stocks for you can get tedious and confusing
Too many things to follow such News/Social Media in order to decide on where to invest 

# Solution
With advancement in ML it has become more possible to model stock prediction
Most current models involve either sentiment analysis or an analysis of history of the stock prices which may not be fully reliable since stock depend on both the news and historic data
My model combines both sentiment analysis of the news along with an analysis on the numerical stats of SP500 to predict future closing prices for stocks
I use both an LSTM(Long-short term memory) model and a GRU (Gating Recruiting Units) model and use a Stacking model as a meta learner to create a final model to combine both models for the best accuracy

#Architecture
Most models rely on either one LSTM model or GRU model, My approach is different since I use both and use an ensemble model to combine both predictions
