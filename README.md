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

# Architecture

<img width="897" alt="Screen Shot 2021-11-21 at 9 18 42 AM" src="https://user-images.githubusercontent.com/37557541/142772338-a638be59-8916-4814-8683-5f7f7ce259d9.png">

Most models rely on either one LSTM model or GRU model, my approach is different since I use both and use an ensemble model to combine both predictions

# iOS App UI

<p float="left">
  <img src="https://user-images.githubusercontent.com/37557541/142772375-c8176659-ef16-4d59-ba4f-9623ec517e92.png" width="400" />
  <img src="https://user-images.githubusercontent.com/37557541/142772380-74cd38de-48ed-4059-96f4-7854c35f024a.png" width="400" />
  <img src="https://user-images.githubusercontent.com/37557541/142772379-a12d0244-ea5d-4d49-9faf-15e524c22600.png" width="400" /> 
</p>
