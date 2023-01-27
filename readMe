This exercise reflects the resolution of a recommender based on a competition carried out by Flow.
With the exercise we practised all the units learn in Data Science course by Ícaro and UNC 2022/23.

Exercise 
3rd  Recommender system

In this exercise we are going to work with a recommendation system.

We are going to take advantage of the Telecom competition that we saw since the data is real and it is good to practice.

In the following repository you can find the meaning of each column of the data sets:
https://github.com/Datathon2021/Recommender 

Work in:

- Divide set into train and test. Take the data until March 1, 2021 as a train. From March 1 onwards, reserve for testing.
- Develop a recommender. The recommender must be able to generate recommendations for ALL users (including cold starts that will not have views on the trainset). Generate 20 recommendations per user.
- The recommendations have to be for each account_id and content_id (NOT asset_id) must be recommended. You can find this in the competitor repository.
- The content they recommend does not have to have been previously seen by users (filter).
- Evaluate it with MAP.

Recommendations:
- In this case we do not have explicit ratings like the cases we saw, you must generate these ratings using some criteria. The simplest might be to use binary ratings (saw it/didn't see it).
- There is a column that tells us until when the content will be available
- The column **end_vod_date**: "end date of the availability of the asset on the platform" can be very useful. Does it make sense to recommend something that is not going to be available in the test set? (as of March 1, 2021).
- Starting with something SIMPLE. Not all columns in the data set are satisfied. They won't need to use all of them, many columns can be dropped depending on the approach they take.
