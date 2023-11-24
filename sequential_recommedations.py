"""
DATA.ML.360-2023-2024-1 - Recommender Systems
Assignment 3 - Sequential Recommendations
Sachini Hewage (152258085) & Robin Ivan Villa Soto (151814365)
November 25, 2023
"""

import matplotlib.pyplot as plt
from group_disagreement import *


def get_sequential_score(alpha, means, leasts):
    """
    This function takes in two dataframes, containing predicted group scores
    by the means and the least misery methods.
    For each item, it calculates a new weighted score taking into account
    the given alpha as a weight for each side.
    :param alpha: float, ranging from 0 to 1, where 1 is the most disagreement
    :param means: df, containing all the mean movie ratings
    :param leasts: df, containing all the least misery movie ratings
    :return: df, containing the updated ratings, sorted
    """
    score_dict = {}
    for i in means.index:
        mn = means.loc[i].values[0]
        lm = leasts.loc[i].values[0]
        newscore = (1 - alpha) * mn + alpha * lm
        score_dict[i] = newscore

    df = pd.DataFrame.from_dict(score_dict, orient='index',
                                columns=['Predicted Value'])
    df.sort_values(by='Predicted Value', inplace=True, ascending=False)

    return df


def get_sat(group_top_10, user_all_ratings):
    """
    This function takes a group's top 10 movies in a dataframe and the
    predicted ratings for all unwatched movies for one user of that group in
    a dataframe and produces the satisfaction score for the user for this
    group recommendations.
    :param user_all_ratings:
    :param group_top_10: df, dataframe of top 10 recommendations
            for a particular user group
    :return: float, the user's satisfaction score for this group recommendation
    """
    score_user_top = sum(user_all_ratings.iloc[0:9]["Predicted Value"])
    group_user_merged = pd.merge(group_top_10, user_all_ratings,
                                 left_index=True, right_index=True,
                                 suffixes=(" Group", " User"))

    score_for_group_top = sum(
        group_user_merged.iloc[0:9]["Predicted Value User"])
    user_satisfaction = score_for_group_top / score_user_top
    return user_satisfaction


def get_alpha(user_rating_dict, grouptop10):
    """
    This function takes in the user ratings and the group top 10, and computes,
    for each user, the satisfaction score. It then calculates alpha as a
    measure of the disagreement in dissatisfaction.
    :param user_rating_dict: dict, containing the ratings for every user
    :param grouptop10: df, containing the top 10 movies recommended
    :return: float, alpha: the level of disagreement from 0 to 1,
    also, the list containing the satisfaction scores
    """
    satisfaction_scores = []

    for i in user_rating_dict:
        ratings = user_rating_dict[i]
        user_satisfaction = get_sat(grouptop10, ratings)
        satisfaction_scores.append(user_satisfaction)

    alpha = max(satisfaction_scores) - min(satisfaction_scores)

    return alpha, satisfaction_scores


def get_user_feedback(round_topn, leastmis, means, group):
    """
    This function does several things:
    - It asks the user what movie they chose, and drops it from the possible
      recommendations pool
    - It asks each member of the group to rate the movie from 1 to 5, and
      computes their selection satisfaction from these scores.
    - It calculates the selection alpha to be used later
    :param round_topn: df, the top n movies suggested in the previous round
    :param leastmis: df, the least misery scores for the movies
    :param means: df, the means scores for the movies
    :param group: list, the members of the group
    :return: float, the selection alpha
    """
    while True:
        user_selection = input("Please enter the movie ID you have selected: ")
        if int(user_selection) in round_topn.index:
            leastmis.drop(int(user_selection), inplace=True)
            means.drop(int(user_selection), inplace=True)

            member_scores = []
            for member in group:
                while True:
                    member_score = float(input(
                        f"Enter rating for user {member} from 0-5 for {user_selection}: "))
                    if 0 <= member_score <= 5:
                        member_scores.append(member_score)
                        break
            alpha_selection = (max(member_scores) - min(member_scores)) / 5

            return alpha_selection
        else:
            print("Invalid movie ID!")


def main():
    # initializing the program
    group = [598, 210, 400]
    number_of_rounds = 3
    recommendations_per_round = 10

    # initializing the dataset containing the movie names
    movies = pd.read_csv("movies.csv", index_col='movieId')

    # creating a dictionary to store the satisfaction scores
    sat_scores = {}
    for user in group:
        sat_scores[user] = []

    # predicted ratings for the individuals
    individual_ratings = {}
    for i in group:
        predicted_ratings = recommend_movies(i, 20, 10000)
        individual_ratings[i] = predicted_ratings

    # predicted group ratings, base case
    preds = improved_group_preds(group)
    mean_preds = mean_rating(preds, len(preds), 'means')
    leastmis_preds = mean_rating(preds, len(preds), 'least_misery')

    roundn_topn = mean_preds.iloc[0:recommendations_per_round]
    roundn_topn_formatted = format_output(roundn_topn, movies)
    print(f"Top {recommendations_per_round} movies for this round:")
    print(roundn_topn_formatted)
    print()

    # calculating the scores to be used and the user satisfaction
    alpha_projection, round_sat = get_alpha(individual_ratings, roundn_topn)
    for i in range(len(round_sat)):
        sat_scores[group[i]].append(round_sat[i])

    # Rounds 2 and 3
    for i in range(1, number_of_rounds):
        alpha_selection = get_user_feedback(roundn_topn, leastmis_preds,
                                            mean_preds, group)
        alpha = (alpha_projection + alpha_selection) / 2
        newscores = get_sequential_score(alpha, mean_preds, leastmis_preds)
        roundn_topn = newscores.iloc[0:recommendations_per_round]
        roundn_topn_formatted = format_output(roundn_topn, movies)

        print()
        print(f"Top {recommendations_per_round} movies for this round:")
        print(roundn_topn_formatted)
        print()

        alpha_projection, round_sat = get_alpha(individual_ratings,
                                                roundn_topn)
        for j in range(len(round_sat)):
            sat_scores[group[j]].append(round_sat[j])

    for user in sat_scores:
        print(f"User {user} satisfaction scores: {sat_scores[user]}")

    ### Plotting Example Results ###
    user1 = sat_scores[group[0]]
    user2 = sat_scores[group[1]]
    user3 = sat_scores[group[2]]

    x = ['Round1', 'Round2', 'Round3']
    x_axis = np.arange(len(x))

    plt.bar(x_axis - 0.3, user1, 0.3, label='User1')
    plt.bar(x_axis, user2, 0.3, label='User2')
    plt.bar(x_axis + 0.3, user3, 0.3, label='User3')

    plt.xticks(x_axis, x)
    plt.ylim(0.7, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Satisfaction")
    plt.title("Satisfaction Score Over Three Iterations")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
