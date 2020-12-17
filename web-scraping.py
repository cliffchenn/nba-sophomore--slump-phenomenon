import pandas as pd
import requests
from bs4 import BeautifulSoup
from secrets import best_urls, best_players, worst_urls, worst_players
import time


def combine_data(url, player):
    diction = {}
    for i in range(len(player)):
        diction[player[i]] = url[i]
    return diction


def scrape_data_rook(info_dict, player):  # scrape rookie stats
    raw = []

    for i in range(len(info_dict)):
        page = requests.get(info_dict[player[i]])
        soup = BeautifulSoup(page.content, "lxml")
        info = soup.find("div", class_="table_outer_container")\
            .find("tr", class_="full_table").find_all("td", class_="right")
        raw.append(info)

    return raw


def scrape_data_soph(info_dict, player):  # scrape sophomore stats
    raw = []

    for i in range(len(info_dict)):
        page = requests.get(info_dict[player[i]])
        soup = BeautifulSoup(page.content, "lxml")
        info = soup.find("div", class_="table_outer_container")\
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
    return df.transpose()


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
    player = "Lowry"  # change player
    compare_dict[player] = "https://www.basketball-reference.com/players/l/lowryky01.html"  # change stat page url

    # find best years
    best_rookie_df = clean_data1(scrape_data_rook(best_player_directory, best_players), best_players)
    best_soph_df = clean_data1(scrape_data_soph(best_player_directory, best_players), best_players)
    delta_best_df = (best_soph_df - best_rookie_df).round(1)

    # find worst years
    worst_rookie_df = clean_data1(scrape_data_rook(worst_player_directory, worst_players), worst_players)
    worst_soph_df = clean_data1(scrape_data_soph(worst_player_directory, worst_players), worst_players)
    delta_worst_df = (worst_soph_df - worst_rookie_df).round(1)

    # store max and min for each
    range_df = find_range(delta_best_df, delta_worst_df).sort_index(ascending=True)
    range_df.columns = [player]

    # finding average
    avg_prog_df = pd.DataFrame(pd.concat([delta_best_df, delta_worst_df], axis=1, sort=False, join="inner")
                               .mean(axis=1), columns=[player]).round(1).sort_index(ascending=True)

    # use averages to normalize values - divide games played by 82
    compare_player = [player]
    compare_df = (clean_data1(scrape_data_soph(compare_dict, compare_player), compare_player) -
                 clean_data1(scrape_data_rook(compare_dict, compare_player), compare_player)).round(1).sort_index(ascending=True)

    # create a single value index
    avg_index = (avg_prog_df / range_df).mean(axis=0)[0]
    player_index = ((compare_df - avg_prog_df) / range_df).mean(axis=0)[0]
    final_score = (player_index - avg_index)/avg_index

    # print the overall percentage difference between average performance and player performance
    # print("\n- - - " + player, end="")
    # print(f" Percent Difference:{(final_score): .2f}% - - -")

    


    print(f"\n- - - Runtime:{(time.time() - start_time): .2f}s - - -")

