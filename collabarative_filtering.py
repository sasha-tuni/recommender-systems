"""
DATA.ML.360-2023-2024-1 - Recommender Systems
Assignment 1 - User-based Collaborative Filtering Recommendations
Sachini Hewage (152258085) & Robin Ivan Villa Soto (151814365)
November 2, 2023

This program uses a matrix of users and their ratings of a series of movies.
For a selected user, it generates a list of the top n most similar users based
on either Pearson's Correlation or a Cosine Similarity Function.
It then takes this set of n similar users and uses it to predict the rating
that the user of interest would give to any movie they haven't watched.
With that information, it outputs a list of the top n movies to recommend
to our user of interest.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_top(user, n, rating, averagedf):
    """
    Gets the top n most similar users to our user of interest.
    Returns a data frame with the user IDs and the correlation scores.
    :param averagedf: df, containing the average ratings for all users
    :param user: str, the user ID of our user of interest
    :param n: int, the number of most similar users we want
    :param rating: df, the processed dataframe with the ratings
    :return: df, a dataframe with the top n most similar users
    """
    # interger location of selected user is at one place before
    # (iloc indexing starts at 0 )
    rated_movies = rating.loc[user].dropna().index.to_list()

    # Initiate a dataframe with the first movie the user has rated
    # as the first column with all users
    new_ratings = rating[rated_movies[0]]

    # Add all movies the user has rated to this dataframe
    for i in range(len(rated_movies)):
        new_ratings = pd.concat([new_ratings, rating[rated_movies[i]]],
                                axis=1)

    # Remove the duplicated first column
    ratings_df = new_ratings.T[~new_ratings.T.index.duplicated(keep='first')].T

    # I do not want truncation
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Extract user1 and selected user overlaps
    corr_dict = {}
    for i in ratings_df.index:
        common = mat_generator(ratings_df, i, user)
        newuser, corr = df_to_corr(common, averagedf)
        corr_dict[newuser] = corr

    top = topn(n, corr_dict, user)
    return top


def mat_generator(ratings_data, current_user, sel_user):
    """
    This function takes the ratings data matrix, and, for a given pair
    of users, extracts the ratings for the movies they both have watched
    :param ratings_data: df, containing ratings
    :param current_user: int, the userID of the user we're comparing with
    :param sel_user: int, the userID of the user of interest
    :return: df, a 2Xn matrix containing both users' ratings
    """
    df_temp = ratings_data.loc[[current_user, sel_user]].dropna(axis=1)
    return df_temp


def df_to_corr(dataframe, avgdf):
    """
    This funtion takes a dataframe with the user IDs
    and the movies they have in common.
    The first row of the dataframe contains info for the new user
    The second row contains info for our user of interest
    It calculates the correlation between the users.
    :param avgdf: df, containing the mean ratings for all users
    :param dataframe: df, contains the userid's and the ratings
    :return: list, of the form [userid2, correlation]
    """

    # Datframe has newuser, interestuser in that order
    newuserid = dataframe.index[0]
    interestuserid = dataframe.index[1]
    ratings1 = dataframe.iloc[0].values
    ratings2 = dataframe.iloc[1].values
    correlation = 0

    mean1 = avgdf.loc[newuserid].values[0]
    mean2 = avgdf.loc[interestuserid].values[0]

    if len(ratings1) == 0 or len(ratings2) == 0:
        return [newuserid, 0]

    correlation_type = 'Cosine'

    if correlation_type == 'Pearson':
        correlation = pearson(ratings1, ratings2, mean1, mean2)
    elif correlation_type == 'Cosine':
        correlation = cosine_sim(ratings1, ratings2)

    vec = [newuserid, correlation]
    return vec


def pearson(person1, person2, mean1, mean2):
    """
    This function calculates the Pearson correlation between 2 users
    who have rated the same movies.
    It requires two vectors of the same size with the scores they have assigned
    :param mean2: fl, the mean rating given by person 2
    :param mean1: fl, the mean rating given by person 1
    :param person1: list, the ratings person 1 gave
    :param person2: list, the ratings person 2 gave
    :return: fl, the Pearson correlation coefficient
    """

    numerator = 0
    den1 = 0
    den2 = 0

    for i in range(0, len(person1)):
        val1 = person1[i] - mean1
        val2 = person2[i] - mean2
        numerator += val1 * val2
        den1 += val1 ** 2
        den2 += val2 ** 2

    denominator = (den1 ** 0.5) * (den2 ** 0.5)
    if denominator == 0:
        return 0

    corr = numerator / denominator

    return corr


def cosine_sim(person1, person2):
    """
    This function calculates the Pearson correlation between 2 users
    who have rated the same movies.
    It requires two vectors of the same size with the scores they have assigned
    :param person1: list, the ratings person 1 gave
    :param person2: list, the ratings person 2 gave
    :return: fl, the Pearson correlation coefficient
    """

    vector = [np.array(person1), np.array(person2)]
    corr = cosine_similarity(vector)[0, 1]

    return corr


def topn(n, dic, user_of_int):
    """
    This takes in a dictionary with the correlations for each user
    It discards the row containing the correlation between the user of interest
    and themselves
    It turns the dictionary to a sortable dataframe
    It sorts the correlations from highest to lowest
    It returns the top n users with the highest correlation
    :param dic: dict, containing the user IDs and the correlations
    :param user_of_int: the user for which we are calculating the corr
    :param n: int, the number of users you want. Default: 10
    :return: a dataframe containing the top 10 correlations
    """

    del dic[user_of_int]
    corrdf = pd.DataFrame.from_dict(dic, orient='index',
                                    columns=['Correlation'])
    sortedcorr = corrdf.sort_values(by='Correlation', ascending=False)
    topcorr = sortedcorr.iloc[0:n]
    return topcorr


def movie_pred(ratings_mat, top_users, avg_matrix, userofint, n):
    """
    This function gives us the top n movies recommended for a user given
    you know the top 10 most similar users
    :param ratings_mat: the ratings of the movies for the similar users
    :param top_users: the top most similar users
    :param avg_matrix: the average ratings for all users
    :param userofint: the user ID of the user of interest
    :param n: the number of movies we want
    :return:
    """

    pred_dict = {}

    interest_mean = avg_matrix.loc[userofint].values[0]

    for movie in ratings_mat.columns:
        peer_ratings = pd.DataFrame(
            ratings_mat[ratings_mat[movie].notna()][movie])
        numerator = 0
        denominator = 0
        curr_prediction = 0

        for i in peer_ratings.index:
            curr_rating = peer_ratings.loc[i][movie]
            curr_sim = top_users.loc[i].values[0]
            curr_mean = avg_matrix.loc[i].values[0]

            numerator += curr_sim * (curr_rating - curr_mean)
            denominator += curr_sim

            curr_prediction = interest_mean + (numerator / denominator)
        pred_dict[movie] = curr_prediction

    moviepred_df = pd.DataFrame.from_dict(pred_dict, orient='index',
                                          columns=['Predicted Value'])
    sorted_pred = moviepred_df.sort_values(by='Predicted Value',
                                           ascending=False)
    toprated = sorted_pred.iloc[0:n]

    return toprated


def format_output(toprated_df, movies):
    top_n_movies = pd.merge(toprated_df, movies,
                            left_index=True, right_index=True)
    del top_n_movies["genres"]
    top_n_movies.loc[top_n_movies["Predicted Value"] > 5,
                     "Predicted Value"] = 5
    return top_n_movies


def main():
    """
    Part 1: Loading and formatting the data
    """
    # Read data
    ratings_temp = pd.read_csv("ratings.csv")
    #print(ratings_temp.head())
    #print(f"This dataset has {len(ratings_temp)} rows")

    # Pivot the table
    ratings = pd.pivot_table(index="userId", columns="movieId",
                             values="rating", data=ratings_temp)

    # Generate a dataframe of averages for all users
    avg_df = pd.DataFrame()
    avg_df['mean'] = ratings.mean(numeric_only=True, axis=1)

    """
    Part 2: For a selected user, getting the top n most similar users
    And the similarity of each with our selected user
    """

    selected_user = 598
    n = 10

    # top_n_users
    top_n_users = get_top(selected_user, n, ratings, avg_df)
    #print(top_n_users)

    """
    Part 3: Recommending movies based on the ratings of the most similar users
    """

    # Get top_n user's indexes
    top_ilocs = top_n_users.index.to_list()

    # Append selected user's index to it
    top_ilocs.append(selected_user)

    # subrtract 1 from each element and make a list so that we can use .iloc
    top_ilocs = (np.array(top_ilocs) - 1).tolist()

    # Extract data for the top_n + selected user
    top_n_df = ratings.iloc[top_ilocs]

    # Drop columns where nobody has rated
    top_n_df = top_n_df.dropna(axis=1, how='all')
    mask = ~(top_n_df.iloc[-1].notna())
    top_n_data = top_n_df.loc[:, mask]

    # Gets the predicted ratings for our user for the top n recommended movies
    n_for_movies = 10
    val = movie_pred(top_n_data, top_n_users, avg_df, selected_user, n_for_movies)
    #print(val)

    # Read movie names to id mapping
    movies = pd.read_csv("movies.csv", index_col='movieId')

    top_n_movies = format_output(val, movies)
    print(top_n_movies)


if __name__ == "__main__":
    main()
