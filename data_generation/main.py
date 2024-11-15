from itertools import compress, starmap

import pandas
import datetime
import geopy
import os
import csv

import pandas as pd
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import boxscoreplayertrackv3
from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import boxscoresummaryv2
from nba_api.stats.endpoints import leaguegamelog
from pandas import DataFrame, read_csv
from geopy import distance
from nba_api.stats.endpoints import leagueleaders

# The code and reasoning for disabling error messages is from
# https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
from warnings import simplefilter

from requests import ReadTimeout

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

nba_id = '00'
# Controls how many players PER TEAM are in each datapoint.
topNPlayerThreshold = 7
# Controls what point differential counts as a close game (INCLUSIVE)
closenessThreshold = 6
# Controls the rest days value that we use for season starting games
firstGameRest = 50
# The timeout threshold for all game related calls
game_timeout = 30
# A custom header to keep stats.nba.api as a keep alive
# From https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/examples.md#endpoint-usage-example
headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

# A dictionary of TeamIds to (lat, lon) pairs
CITY_LOCATIONS = {1610612737: (33.7501, -84.3885), 1610612738: (42.3601, -71.0589), 1610612739: (41.4993, -81.6944),
                  1610612740: (29.9509, -90.0758), 1610612741: (41.8781, -87.6298), 1610612742: (32.7767, -96.7970),
                  1610612743: (39.7392, -104.9903), 1610612744: (37.7749, -122.4194), 1610612745: (29.7601, -95.3701),
                  1610612746: (34.0549, -118.2426), 1610612747: (34.0549, -118.2426), 1610612748: (25.7617, -80.1918),
                  1610612749: (43.0389, -87.9065), 1610612750: (44.9778, -93.2650), 1610612751: (40.6782, -73.9442),
                  1610612752: (40.7505, -73.9934), 1610612753: (28.5384, -81.3789), 1610612754: (39.7691, -86.1580),
                  1610612755: (39.9526, -75.1652), 1610612756: (33.4484, -112.0740), 1610612757: (45.5152, -122.6784),
                  1610612758: (38.5781, -121.4944), 1610612759: (29.4252, -98.4946), 1610612760: (35.4676, -97.5164),
                  1610612761: (43.6532, -79.3832), 1610612762: (30.7608, -11.8910), 1610612763: (35.1495, -90.0490),
                  1610612764: (38.9072, -77.0369), 1610612765: (42.3314, -83.0458), 1610612766: (35.2271, -80.8431)}


# TODO: Find a way to deal with midseason coaching changes.

# https://github.com/swar/nba_api/blob/master/docs/nba_api/live/endpoints/boxscore.md: Officials database

# Returns an integer representing the given MM:SS format in seconds
def minutes_to_seconds(minutes: str):
    minutes_seconds = minutes.split(":")
    return int(minutes_seconds[0]) * 60 + int(minutes_seconds[1])


# Returns the topNPlayers who played for the given team in the given game, sorted by minutes played
def query_top_players_on_team(player_game_data: DataFrame, team_id):
    return player_game_data.query('teamId == @team_id') \
        .sort_values('minutes', ascending=False, key=(lambda minutes: minutes.map(minutes_to_seconds))) \
        .head(topNPlayerThreshold)


# Transforms the minutes column into seconds to change it into a column.
def generate_player_stats(player_data):
    player_data.insert(0, 'seconds', player_data['minutes'].apply(minutes_to_seconds))
    return player_data.drop(['minutes', 'teamId', 'personId'], axis=1)


# Determines if the given player played in the game.
def is_player_present(game_box_score, player_id):
    return (game_box_score['personId'] == player_id).any()


# Given a game box score, generates a vector for both teams of the top N players (and their stats) on this team and
# the opposing team
def generate_game_stats_per_team(game_box_score, star_players_id):
    # Filter players to get top players for home and away teams based on teamId
    home_team_top_players = query_top_players_on_team(game_box_score, game_box_score.iloc[0]['teamId'])
    away_team_top_players = query_top_players_on_team(game_box_score, game_box_score.iloc[-1]['teamId'])

    # Get team IDs directly from these DataFrames
    home_team_id = home_team_top_players.iloc[0]['teamId']
    away_team_id = away_team_top_players.iloc[0]['teamId']

    # Remove extra information form the players (like teamId and personId)
    home_team_top_players = generate_player_stats(home_team_top_players)
    away_team_top_players = generate_player_stats(away_team_top_players)

    # Collect player data for each team without adding teamId in individual dictionaries
    home_team_data = [{'STAR_PLAYER_PRESENT': [is_player_present(game_box_score, star_players_id[home_team_id])]}]
    away_team_data = [{'STAR_PLAYER_PRESENT': [is_player_present(game_box_score, star_players_id[away_team_id])]}]

    rank = 1
    for _, home_team_player in home_team_top_players.iterrows():
        player_data = {col + f'_my_player_{rank}': [val] for col, val in zip(home_team_player.index, home_team_player.values) if col != 'teamId'}
        home_team_data.append(player_data)
        
        opponent_data = {col + f'_opposing_player_{rank}': [val] for col, val in zip(home_team_player.index, home_team_player.values) if col != 'teamId'}
        away_team_data.append(opponent_data)
        rank += 1

    rank = 1
    for _, away_team_player in away_team_top_players.iterrows():
        opponent_data = {col + f'_opposing_player_{rank}': [val] for col, val in zip(away_team_player.index, away_team_player.values) if col != 'teamId'}
        home_team_data.append(opponent_data)
        
        player_data = {col + f'_my_player_{rank}': [val] for col, val in zip(away_team_player.index, away_team_player.values) if col != 'teamId'}
        away_team_data.append(player_data)
        rank += 1

    # Convert collected data to DataFrames
    home_team_players_info = pd.concat([pd.DataFrame(data) for data in home_team_data], axis=1)
    away_team_players_info = pd.concat([pd.DataFrame(data) for data in away_team_data], axis=1)

    # Add `teamId` column at the end of each DataFrame
    home_team_players_info['teamId'] = home_team_id
    away_team_players_info['teamId'] = away_team_id

    return home_team_players_info, away_team_players_info



# Adds the team_id of the home team to the given game_stats as the game's location.
def add_game_location(game_stats, home_team_id):
    game_stats.at[0, 'GAME_LOCATION'] = home_team_id
    return game_stats


# Adds the GAME_ID, scoring, W/L, and closeness information to each team's output vector from
# generate_game_stats_per_team. Both vectors must be passed in in-order to determine which team won/lost.
def add_scoring_statistics(home_team_players_info, away_team_players_info, scores):
    non_player_info = scores[['GAME_ID', 'TEAM_ID', 'PTS', 'TEAM_WINS_LOSSES', 'GAME_DATE_EST']]
    home_team_result = pd.merge(home_team_players_info, non_player_info, left_on='teamId', right_on='TEAM_ID').copy()
    away_team_result = pd.merge(away_team_players_info, non_player_info, left_on='teamId', right_on='TEAM_ID').copy()

    # calculate close game and determine winner
    was_game_close = abs(home_team_result['PTS'] - away_team_result['PTS']) <= closenessThreshold
    home_team_result['CLOSE'] = was_game_close
    away_team_result['CLOSE'] = was_game_close
    home_team_result['WON'] = home_team_result['PTS'] > away_team_result['PTS']
    away_team_result['WON'] = away_team_result['PTS'] > home_team_result['PTS']
    
    return home_team_result, away_team_result


# Calculates the avg number of fouls called on the home team more than the away team.
def generate_ref_statistic(ref_name, season_referee_stats):
    # TODO: Save the index so we don't compute it twice
    if not (season_referee_stats['Info', 'Referee'] == ref_name).any():
        #
        raise RuntimeError(f"Could not find a referee named {ref_name}!")
        # If the referee is not found, return 'N/A' or another placeholder value
        return 'N/A'
    return season_referee_stats[season_referee_stats['Info', 'Referee'] == ref_name]\
        ['Home Minus Visitor', 'PF'].values[0]


# Generates the average extra personal fouls per game the referees called on the home team versus the away team.
# A negative number means the refs called less fouls on the home team than the away team
def generate_refs_statistics(game_officials, season_referee_stats):
    home_pf_bias = 0
    # Loop through each referee, fetching the bias or using a default if not available
    for _, row in game_officials[['FIRST_NAME', 'LAST_NAME']].iterrows():
        ref_name = f"{row['FIRST_NAME']} {row['LAST_NAME']}"
        ref_bias = generate_ref_statistic(ref_name, season_referee_stats)
        # Only add to the bias if it's a numeric value
        if ref_bias != 'N/A':
            home_pf_bias += ref_bias
    return home_pf_bias


# Returns a 2 element list of the home team and away team's information.
def pull_game_data(game_id, season_referee_stats, star_player_ids):
    # 2 length list, player stats are the first element.
    players_stats = boxscoreplayertrackv3 \
        .BoxScorePlayerTrackV3(game_id=game_id, timeout=game_timeout, headers=headers).get_data_frames()
    game_box_score = players_stats[0][
        ['teamId', 'personId', 'minutes', 'speed', 'distance', 'reboundChancesOffensive',
         'reboundChancesTotal', 'touches', 'secondaryAssists', 'freeThrowAssists', 'passes',
         'assists', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted',
         'contestedFieldGoalPercentage', 'uncontestedFieldGoalsMade',
         'uncontestedFieldGoalsAttempted', 'uncontestedFieldGoalsPercentage', 'fieldGoalPercentage',
         'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted',
         'defendedAtRimFieldGoalPercentage']]
    summary = boxscoresummaryv2 \
        .BoxScoreSummaryV2(game_id=game_id, timeout=game_timeout, headers=headers).get_data_frames()
    home_location = summary[0]['HOME_TEAM_ID'].values[0]
    referees = summary[2]
    scores = summary[5]
    home_team_players_info, away_team_players_info = generate_game_stats_per_team(game_box_score, star_player_ids)
    ref_home_bias = generate_refs_statistics(referees, season_referee_stats)
    home_team_players_info['REF_BIAS'] = ref_home_bias
    away_team_players_info['REF_BIAS'] = -1 * ref_home_bias
    home_team_result, away_team_result = add_scoring_statistics(home_team_players_info, away_team_players_info, scores)
    home_team_result = add_game_location(home_team_result, home_location)
    away_team_result = add_game_location(away_team_result, home_location)
    return [home_team_result, away_team_result]


def distance_between_teams(team_id_1, team_id_2):
    location_1 = CITY_LOCATIONS[team_id_1]
    location_2 = CITY_LOCATIONS[team_id_2]
    return distance.distance(location_1, location_2).miles


# This function is based off of the following stack overflow answer: https://stackoverflow.com/a/65313188
def starfilter(pred, values):
    return list(compress(values, starmap(pred, values)))


# Updates the record of the recent games played (games played in the last 7 days) given the latest game resu;t
def update_recent_games(recent_games, latest_game_date, latest_game_result):
    new_recent_games = starfilter(lambda date, _: (date - latest_game_date).days < 7, recent_games)
    new_recent_games.append((latest_game_date, latest_game_result))
    return new_recent_games


def recent_games_win_pct(recent_games):
    return 0 if len(recent_games) == 0 else len(starfilter(lambda _, won: won, recent_games)) / len(recent_games)


# teams ONLY have 1 coach per season, so this is a major issue to work through. Currenlty, I am assuming that there
# is only one coach per year, and calculating a team's win percentage and close win percentage
def pull_team_data(game_data: DataFrame, team_id):
    team_games = game_data[game_data['teamId'] == team_id].copy()

    # Initialize lists to store column data 
    win_pct_list = []
    close_win_pct_list = []
    is_back_to_back_list = []
    rest_days_list = []
    distance_list = []
    recent_win_pct_list = []

    # Counters/placeholders initialization
    wins = 0
    close_wins = 0
    games_played = 0
    close_games_played = 0
    previous_game_date = None  # Start with None to handle the first game separately
    previous_game_location = team_id
    # Sliding window of (date, result) tuples representing the team's recent games.
    recent_games = []

    for index, row in team_games.iterrows():
        # Calculate win percentages
        win_pct = 0 if games_played == 0 else wins / games_played
        close_win_pct = 0 if close_games_played == 0 else close_wins / close_games_played
        win_pct_list.append(win_pct)
        close_win_pct_list.append(close_win_pct)

        # Calculate rest days and back-to-back status
        current_game_date = pd.to_datetime(row['GAME_DATE_EST'], format='%Y-%m-%dT%H:%M:%S')
        
        if previous_game_date is None:
            # First game of the season
            rest_days = firstGameRest
        else:
            # Calculate the rest days as the difference from the previous game date
            rest_days = (current_game_date - previous_game_date).days - 1
        
        is_back_to_back = rest_days == 0
        is_back_to_back_list.append(is_back_to_back)
        rest_days_list.append(rest_days)

        # Calculate travel distance
        distance = distance_between_teams(previous_game_location, row['GAME_LOCATION'])
        distance_list.append(distance)
        previous_game_location = row['GAME_LOCATION']

        # Calculate recent win pct
        recent_win_pct = recent_games_win_pct(recent_games)
        recent_win_pct_list.append(recent_win_pct)
        recent_games = update_recent_games(recent_games, current_game_date, row['WON'])

        # Update previous_game_date to current date for the next iteration
        previous_game_date = current_game_date

        # Update win and close win counts
        wins += int(row['WON'])
        games_played += 1
        if row['CLOSE']:
            close_wins += int(row['WON'])
            close_games_played += 1

    # Add calculated lists to the DataFrame
    team_games['WIN_PCT'] = win_pct_list
    team_games['CLOSE_WIN_PCT'] = close_win_pct_list
    team_games['IS_BACK_TO_BACK'] = is_back_to_back_list
    team_games['REST_DAYS'] = rest_days_list
    team_games['DISTANCE'] = distance_list
    team_games['RECENT_WIN_PCT'] = recent_win_pct_list

    return team_games


# Loads and returns the stats for the referees from the given season.
def load_referees(season: str):
    return read_csv(f'referee_data/{season}.csv', header=[0, 1])


# Updates each game to add the opponent recent win percentage, and TODO: whether the star player was present
def update_game_stats(game_id, all_game_stats):
    game_stats = all_game_stats[all_game_stats['GAME_ID'] == game_id]
    # .values needed here to remove the index information from the resulting series
    game_stats.insert(len(game_stats.columns), 'OPPONENT_WIN_PCT', game_stats['RECENT_WIN_PCT'][::-1].values,
                      allow_duplicates=False)
    # Project away all extra information needed for calculating features
    return game_stats.drop(['GAME_ID', 'GAME_DATE_EST'], axis=1)


# Given a list of team_ids generates a corresponding map of teams to star players.
def generate_star_players(team_ids: list, season: str):
    star_players = dict()
    best_players = leagueleaders.LeagueLeaders(league_id='00', per_mode48='PerGame', scope='S', season=season,
                                               season_type_all_star='Regular Season',
                                               headers=headers).get_data_frames()[0] \
        .sort_values('EFF', ascending=False)
    for team_id in team_ids:
        star_players[team_id] = best_players[best_players['TEAM_ID'] == team_id].head(1)['PLAYER_ID'].values[0]
    return star_players


# Generates the game stats for the given season. If we are getting rate limited, writes the output to disk
# before crashing. If the game stats for a season already exist, loads them up and continues going.
def generate_game_stats_for_season(season: str, game_ids, season_referee_stats, star_player_ids):
    cached_stats_path = f"game_stats/{season}.csv"
    season_games_data = DataFrame()
    remaining_ids = game_ids
    if os.path.exists(cached_stats_path):
        season_games_data = read_csv(cached_stats_path, dtype={'GAME_ID': str})
        remaining_ids = set(game_ids) - set(season_games_data['GAME_ID'].values)
    for game_id in remaining_ids:
            try:
                game_results = pull_game_data(game_id, season_referee_stats, star_player_ids)
                season_games_data = pandas.concat([season_games_data] + game_results, axis=0, ignore_index=True)
            # If an error occurs, save where we're at to resume later.
            except Exception as e:
                # Cache the accumulated data before rethrowing the exception
                season_games_data.to_csv(cached_stats_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                # Rethrow the exception to further unwind the stack
                raise e
    # successfully have gotten the stats for every game. Cache it in case later processing steps fail.
    season_games_data.to_csv(cached_stats_path, index=False)
    return season_games_data


# Generates the regular season statistics for every game in a season.
def generate_season_stats(season: str):
    # This is only for testing to test on a smaller dataset.
    game_limit = 10
    referees = load_referees(season)
    season_games = leaguegamelog.LeagueGameLog(counter=10, direction="ASC", league_id=nba_id,
                                               season=season, season_type_all_star='Regular Season',
                                               sorter='DATE', headers=headers).get_data_frames()[0]
    all_team_ids = season_games['TEAM_ID'].unique()
    star_player_ids = generate_star_players(all_team_ids, season)
    all_game_ids = season_games['GAME_ID'].unique()
    # monstrous list comprehension because there's no builtin flatten function for python lists.
    all_game_stats = generate_game_stats_for_season(season, all_game_ids, referees, star_player_ids)
    all_game_stats_with_team_stats = \
        pandas.concat([pull_team_data(all_game_stats, team_id) for team_id in all_team_ids],
                      axis=0,
                      ignore_index=True)
    complete_season_stats_list = [update_game_stats(game_id, all_game_stats_with_team_stats)
                                  for game_id in all_game_ids]
    complete_season_stats = pandas.concat(complete_season_stats_list, axis=0, ignore_index=True)
    # Drops the last non linear features (teamId and TEAM_WIN_LOSSES).
    return complete_season_stats.drop(['teamId', 'TEAM_WINS_LOSSES'], axis=1)


# Generates the given season and writes the data to the corresponding csv in output_data.
# In case an error occurs, the data for all the games that have been processed is written to game_stats.
def write_season_stats(season):
    season_stats = generate_season_stats(season)
    # Solution from https://stackoverflow.com/questions/17383094/how-can-i-map-true-false-to-1-0-in-a-pandas-dataframe
    season_stats.loc[:, ['CLOSE', 'WON', 'IS_BACK_TO_BACK', 'STAR_PLAYER_PRESENT']] = \
        season_stats[['CLOSE', 'WON', 'IS_BACK_TO_BACK', 'STAR_PLAYER_PRESENT']].astype(int)
    season_stats.to_csv(f"output_data/{season}_data.csv", index=False)


# Instructions to run:
# Replace the season in write_season_stats and the program will generate the stats per game in the output_data.
# There most likely will be a timeout error while running the program. The program will cache the season data
# to the appropriate file in game_stats before exiting. Wait approximately a minute or so to avoid getting
# blocked on nba_api and re-run the program. It will load the cached data and resume from there.
if __name__ == '__main__':
    write_season_stats('2021-22')
    # TODO: Figure out how to get coaches on a more granular level than season, since mid season changes SHOULD be
    #  reflected if we decide to use coaches
    teams_coaches = commonteamroster.CommonTeamRoster(team_id='1610612749', season='2023').get_data_frames()
    print(teams.get_teams())
    print(teams_coaches[1][['SEASON', 'FIRST_NAME', 'LAST_NAME', 'COACH_NAME', 'IS_ASSISTANT']])
