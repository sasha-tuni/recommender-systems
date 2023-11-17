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


def mean_rating(user_dataframe, n, kind):
    """
    This function takes in a dataframe containing the predicted movie rating
    for all users, and gets the mean score for each movie. Then, it sorts the
    movies by mean rating, and returns the top n movies.
    :param kind: 'means' or 'least_misery'
    :param n: the number of movie recommendations we want.
    :param user_dataframe: df, containing the ratings for each user
    :return: df, containing the mean rating
    """

    if kind == 'means':
        rec_df = user_dataframe.mean(axis=1)
    elif kind == 'least_misery':
        rec_df = user_dataframe.min(axis=1)
    else:
        print("Type must be 'means' or 'least_misery'")
        return None

    sorted_recs = rec_df.sort_values(ascending=False)
    top_recs = sorted_recs.iloc[0:n]
    top_recs_df = top_recs.to_frame(name='Predicted Value')

    return top_recs_df


def calc_difference(grouplist, method='means'):
    """
    This function takes in a list with the rating that the group gave for all
    the movies (containing only the movies with no missing values). This
    generates a group ranking of the movies using the mean method.
    For each user, it generates a top 10 list, and compares the position of
    these top 10 movies in the user list and the group list, calculating the
    disagreement between these two according to the Spearman's Footrule
    method. It adds up the disagreement for each user, and returns the full
    disagreement score for the group.
    :param method: use "means" or "least_misery"
    :param grouplist: df, containing the group ratings for all the movies
    :return: int, Spearman's Footrule distance for the two lists
    """
    colnames = grouplist.columns
    group_ranking = mean_rating(grouplist, len(grouplist), method)
    group_list = group_ranking.index.tolist()

    disagreement = 0
    disagreements = []

    for i in range(len(grouplist.columns)):
        sorted_group = grouplist.sort_values(by=colnames[i], ascending=False)
        user_movies = sorted_group.index.tolist()
        top = user_movies[0:10]

        user_position = 1
        user_disagreement = 0

        for j in top:
            group_position = group_list.index(j) + 1
            disagree_i = abs(group_position - user_position)
            user_disagreement += disagree_i
            user_position += 1

        disagreements.append(user_disagreement)

    return disagreements

def improved_group_dataframe(users_df):
    """
    This function takes in a dataframe of unwatched movies for the selected users with non-null ratings and
    filters out any movie that has a rating below the user's average.
    :param users_df: df, dataframe of unwatched movies for the selected users with null values removed
    :return: df, movies with above average rating for all users
    """
    for column in users_df.columns:
        mean_of_user = np.mean(users_df[column])
        users_df.loc[:, column] = np.where(users_df[column] < mean_of_user,
                                           np.nan, users_df[column])
    new_group_df = users_df.dropna()
    return new_group_df


def main():
    user_list = [598, 210, 400]
    group_df = group_dataframe(user_list, 20, 10000)
    group_no_na = group_df.dropna()

    improved_group_predictions = improved_group_dataframe(group_no_na)

    movies = pd.read_csv("movies.csv", index_col='movieId')

    improved_means = mean_rating(improved_group_predictions,10,'means')
    improved_least_mis = mean_rating(improved_group_predictions, 10, 'least_misery')
    improved_means_formatted = format_output(improved_means, movies)
    improved_leastmis_formatted = format_output(improved_least_mis, movies)

    print("The top 10 recommended movies using the means method:")
    print(improved_means_formatted)
    print("The top 10 recommended movies using the least misery method:")
    print(improved_leastmis_formatted)
    print()

    diff_basic_mean = calc_difference(group_no_na)
    print(f"The total disagreement (using means) with no adjustment is {diff_basic_mean}")

    diff_improved_mean = calc_difference(improved_group_predictions)
    print(f"The total disagreement (using means) after adjustment is {diff_improved_mean}")
    print()

    diff_basic_leastmis = calc_difference(group_no_na, 'least_misery')
    print(f"The total disagreement (using least misery) with no adjustment is {diff_basic_leastmis}")

    diff_improved_leastmis = calc_difference(improved_group_predictions,'least_misery')
    print(f"The total disagreement (using least misery) after adjustment is {diff_improved_leastmis}")


if __name__ == "__main__":
    main()
