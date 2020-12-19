import pandas as pd
import requests
from bs4 import BeautifulSoup
from secrets import best_urls, best_players, worst_urls, worst_players
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



def combine_data(url, player):
    diction = {}
    for i in range(len(player)):
        diction[player[i]] = url[i]
    return diction


def scrape_data_rook(info_dict, player):  # scrape rookie stats
    raw = []

    for i in range(len(info_dict)):
        page = requests.get(info_dict[player[i]])
        soup = BeautifulSoup(page.content, "html.parser")
        try:
            info = soup.find("div", class_="table_outer_container").find("tr", class_="full_table").find_all("td", class_="right")
        except:
            info = soup.find_all("div", class_="table_outer_container")[1].find("tr", class_="full_table").find_all("td", class_="right")

        raw.append(info)
    return raw


def scrape_data_soph(info_dict, player):  # scrape sophomore stats
    raw = []

    for i in range(len(info_dict)):
        page = requests.get(info_dict[player[i]])
        soup = BeautifulSoup(page.content, "lxml")
        try:
            info = soup.find("div", class_="table_outer_container")\
                .find_all("tr", class_="full_table")[1].find_all("td", class_="right")
        except:
            info = soup.find_all("div", class_="table_outer_container")[1]\
                .find_all("tr", class_="full_table")[1].find_all("td", class_="right")
        raw.append(info)

    return raw


def clean_data1(player_info, rank):
    temp_dict = {}
    categories = []
    for index, player_stat in enumerate(player_info):
        temp_list = []
        for categ in player_stat:
            categ = str(categ).replace('<td class="right" data-stat=', "")
            categ = str(categ).replace('<td class="right iz" data-stat=', "")
            categ = str(categ).replace('</td>', "").replace('>', "").replace('_per_g', "")
            categ = str(categ).replace('<strong', "").replace('</strong', "").replace("''", "")

            if categ.split('"')[2] == "":
                categ = categ + "0.0"

            temp_list.append((float(categ.split('"')[2])))

            if len(categories) < len(player_stat):
                categories.append((categ.split('"'))[1])

        temp_dict[rank[index]] = temp_list

    df = pd.DataFrame.from_dict(temp_dict).transpose()
    df.columns = [categories]

    return df


def is_slump(rook_df, soph_df, slump):
    delta_df = soph_df - rook_df
    if slump:
        delta_df["slump"] = True
    else:
        delta_df["slump"] = False

    return delta_df


def find_range(best, worst):
    range_diff = pd.DataFrame(best.max(axis=1) - worst.min(axis=1), columns=["range"]).round(1)
    for i, value in enumerate(range_diff.range):
        if value == 0.0:
            range_diff["range"][i] = 1.0
    return range_diff


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option('display.max_columns', 500)
    best_player_directory = combine_data(best_urls, best_players)
    worst_player_directory = combine_data(worst_urls, worst_players)

    # determine which player to compare
    compare_dict = {}
    player = "Shane Battier"  # change player
    compare_dict[player] = "https://www.basketball-reference.com/players/b/battish01.html"  # change stat page url

    # find best years
    best_rookie_df = clean_data1(scrape_data_rook(best_player_directory, best_players), best_players)
    best_soph_df = clean_data1(scrape_data_soph(best_player_directory, best_players), best_players)
    delta_best_df = (best_soph_df - best_rookie_df).transpose().round(1)
    delta_best_slump = is_slump(best_rookie_df, best_soph_df, False)

    # find worst years
    worst_rookie_df = clean_data1(scrape_data_rook(worst_player_directory, worst_players), worst_players)
    worst_soph_df = clean_data1(scrape_data_soph(worst_player_directory, worst_players), worst_players)
    delta_worst_df = (worst_soph_df - worst_rookie_df).transpose().round(1)
    delta_worst_slump = is_slump(worst_rookie_df, worst_soph_df, True)

    # store max and min for each
    range_df = find_range(delta_best_df, delta_worst_df).sort_index(ascending=True)
    range_df.columns = [player]

    # finding average
    avg_prog_df = pd.DataFrame(pd.concat([delta_best_df, delta_worst_df], axis=1, sort=False, join="inner")
                               .mean(axis=1), columns=[player]).round(1)

    # use averages to normalize values
    compare_player = [player]
    compare_df = (clean_data1(scrape_data_soph(compare_dict, compare_player), compare_player) -
                 clean_data1(scrape_data_rook(compare_dict, compare_player), compare_player)).round(1).transpose()

    # create a single value index
    avg_index = (avg_prog_df / range_df).mean(axis=0)[0]
    player_index = ((compare_df - avg_prog_df) / range_df).mean(axis=0)[0]
    final_score = (player_index - avg_index)/avg_index

    # print the overall percentage difference between average performance and player performance
    print("\n- - - " + player, end="")
    print(f" Percent Difference: {(final_score): .2f}% - - -")

    # Visualizing Improvement Indices
    indices = pd.read_csv("player_indices.csv", names=["player", "%_change"])

    plt.figure(figsize=(15, 10))
    plt.bar(indices["player"], indices["%_change"], align="center", width=0.8, edgecolor="black", linewidth=2)
    plt.title("Percent Improvement Between Rookie and Sophomore Year")
    plt.xticks(indices["player"], rotation=20)
    plt.style.use("ggplot")
    plt.xlabel("Player")
    plt.ylabel("Percent Change (%)")
    plt.grid(axis="y")

    # slump_df = pd.DataFrame(pd.concat([delta_best_slump, delta_worst_slump]))
    slump_df = pd.concat([delta_best_slump, delta_worst_slump], axis=0)
    # print("SLUMP: \n", slump_df)

    # logistic regression
    # define X, y for dataset
    X = np.asarray(slump_df.drop(columns="slump"))
    y = np.asarray(slump_df[slump_df.columns[len(slump_df.columns) - 1]])

    # normalize dataset
    X = preprocessing.StandardScaler().fit(X).transform(X)

    # model
    LR = LogisticRegression(C=0.01, solver="liblinear").fit(X, y)

    # predict using chosen player
    y_hat = LR.predict(np.asarray(compare_df.transpose()))
    print("The Player Experienced the Sophomore Slump: ", y_hat[0])

    # append player, player index, slump to csv
    player_store = pd.DataFrame({player: [final_score, y_hat[0]]}).transpose()
    player_store.to_csv("player_indices.csv", mode="a", header=False)

    print(f"\n- - - Runtime:{(time.time() - start_time): .2f}s - - -")

    # plt.show()