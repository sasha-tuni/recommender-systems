"""
DATA.ML.360-2023-2024-1 - Recommender Systems
Assignment 2 - Group Recommendations
Sachini Hewage (152258085) & Robin Ivan Villa Soto (151814365)
November 18, 2023
"""

from collabarative_filtering import *
import pandas as pd


def user_ratings(user_id, user_df):
    user_df = user_df.rename(columns={'Predicted Value': f'{user_id}'})
    user_df.sort_index(inplace=True)
    return user_df


def group_dataframe(user_list, no_of_sim, no_top_movies):
    user_df = recommend_movies(user_list[0], no_of_sim, no_top_movies)
    group_pred = user_ratings(user_list[0], user_df)

    for user in user_list[1:]:
        user_df = recommend_movies(user, no_of_sim, no_top_movies)
        user_predictions = user_ratings(user, user_df)
        group_pred = pd.merge(group_pred, user_predictions, how="outer",
                              left_index=True, right_index=True)
    return group_pred


def mean_rating(user_dataframe, n, type):
    """
    This function takes in a dataframe containing the predicted movie rating
    for all users, and gets the mean score for each movie. Then, it sorts the
    movies by mean rating, and returns the top n movies.
    :param type: 'means' or 'least_misery'
    :param n: the number of movie recommendations we want.
    :param user_dataframe: df, containing the ratings for each user
    :return: df, containing the mean rating
    """

    if type == 'means':
        rec_df = user_dataframe.mean(axis=1)
    elif type == 'least_misery':
        rec_df = user_dataframe.min(axis=1)
    else:
        print("Type must be 'means' or 'least_misery'")
        return None

    sorted_recs = rec_df.sort_values(ascending=False)
    top_recs = sorted_recs.iloc[0:n]
    top_recs_df = top_recs.to_frame(name='Predicted Value')

    return top_recs_df


def main():
    user_list = [598, 210, 400]
    group_df = group_dataframe(user_list, 20, 10000)
    group_no_na = group_df.dropna()

    group_prediction_means = mean_rating(group_no_na,10,'means')
    group_prediction_least_mis = mean_rating(group_no_na, 10, 'least_misery')

    movies = pd.read_csv("movies.csv", index_col='movieId')

    means_formatted = format_output(group_prediction_means,movies)
    least_mis_formatted = format_output(group_prediction_least_mis,movies)

    print("The top 10 recommended movies using the means method:")
    print(means_formatted)
    print("The top 10 recommended movies using the least misery method:")
    print(least_mis_formatted)


if __name__ == "__main__":
    main()
