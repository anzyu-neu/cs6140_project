from itertools import compress, starmap

import pandas
import datetime
import geopy

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
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Top level idea:
# Step 1: Create a data frame of the per-game stats. E.g. Player stats, referees, our label points,
# etc. pull_game_data is responsible for doing this (referee stuff is not implemented yet since that should be
# straightforward but annoying). The topNPlayersAvg and other features would go in here. The current 433 features
# that are present are just a placeholder until we figure out which ones we want. Defense would also go here as well.
# Step 2: On the dataset containing all the individual game data gathered in step 1, create the aggregated features,
# e.g. IsBackToBack, RestDays, OppWinPct, and Dist. Features that require looking at a sequence of games.
# We currently have an issue that coaches are recorded on a per season NOT per game basis, meaning in season coaching
# changes will be hard if not impossible to measure.

nba_id = '00'
# Controls how many players PER TEAM are in each datapoint.
topNPlayerThreshold = 7
# Controls what point differential counts as a close game (INCLUSIVE)
closenessThreshold = 6
# Controls the rest days value that we use for season starting games
firstGameRest = 50
# The timeout threshold for all game related calls
game_timeout = 90
# A custom header to keep stats.nba.api as a keep alive
# From https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/examples.md#endpoint-usage-example

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


def generate_player_stats(player_data):
    player_data.insert(0, 'seconds', player_data['minutes'].apply(minutes_to_seconds))
    return player_data.drop(['minutes', 'teamId', 'personId'], axis=1)


# Determines if the given player played in the game.
def is_player_present(game_box_score, player_id):
    return (game_box_score['personId'] == player_id).any()


# Given a game box score, generates a vector for both teams of the top N players (and their stats) on this team and
# the opposing team, including if the star player is available.
def generate_game_stats_per_team(game_box_score, star_player_info):
    team_ids = list(set(game_box_score['teamId']))
    home_team_top_players = generate_player_stats(query_top_players_on_team(game_box_score, team_ids[0]))
    away_team_top_players = generate_player_stats(query_top_players_on_team(game_box_score, team_ids[1]))
    home_team_players_info = DataFrame({'teamId': [team_ids[0]], 'STAR_PLAYER_PRESENT':
                                        is_player_present(game_box_score, star_player_info[team_ids[0]])})
    away_team_players_info = DataFrame({'teamId': [team_ids[1]], 'STAR_PLAYER_PRESENT':
                                        is_player_present(game_box_score, star_player_info[team_ids[1]])})
    rank = 1
    for (_, home_team_player) in home_team_top_players.iterrows():
        home_team_column_names = home_team_player.index.map(lambda col_name: col_name + '_my_player_' + str(rank))
        home_team_players_info[home_team_column_names] = home_team_player.values
        away_team_column_names = home_team_player.index.map(lambda col_name: col_name + '_opposing_player_' + str(rank))
        away_team_players_info[away_team_column_names] = home_team_player.values
        rank += 1
    rank = 1
    for (_, away_team_player) in away_team_top_players.iterrows():
        home_team_column_names = away_team_player.index.map(lambda col_name: col_name + '_opposing_player_' + str(rank))
        home_team_players_info[home_team_column_names] = away_team_player.values
        away_team_column_names = away_team_player.index.map(lambda col_name: col_name + '_my_player_' + str(rank))
        away_team_players_info[away_team_column_names] = away_team_player.values
        rank += 1
    return home_team_players_info.copy(), away_team_players_info.copy()


# Adds the team_id of the home team to the given game_stats as the game's location.
def add_game_location(game_stats, home_team_id):
    game_stats.at[0, 'GAME_LOCATION'] = home_team_id
    return game_stats


# Adds the scoring, W/L, and closeness information to each team's output vector from generate_game_stats_per_team.
# Both vectors must be passed in in order to determine which team won/lost.
def add_scoring_statistics(home_team_players_info, away_team_players_info, scores):
    non_player_info = scores[['GAME_ID', 'TEAM_ID', 'PTS', 'TEAM_WINS_LOSSES', 'GAME_DATE_EST']]
    home_team_result = DataFrame.merge(home_team_players_info, non_player_info, left_on='teamId', right_on='TEAM_ID')
    away_team_result = DataFrame.merge(away_team_players_info, non_player_info, left_on='teamId', right_on='TEAM_ID')
    was_game_close = abs(home_team_result['PTS'] - away_team_result['PTS']) <= closenessThreshold
    home_team_result['CLOSE'] = was_game_close
    away_team_result['CLOSE'] = was_game_close
    home_team_result['WON'] = home_team_result['PTS'] > away_team_result['PTS']
    away_team_result['WON'] = away_team_result['PTS'] > home_team_result['PTS']
    return home_team_result.copy(), away_team_result.copy()


# Calculates the avg number of fouls called on the home team more than the away team.
def generate_ref_statistic(ref_name, season_referee_stats):
    # TODO: Save the index so we don't compute it twice
    if not (season_referee_stats['Info', 'Referee'] == ref_name).any():
        raise RuntimeError(f"Referee {ref_name} not found")
    return season_referee_stats[season_referee_stats['Info', 'Referee'] == ref_name] \
        ['Home Minus Visitor', 'PF'].values[0]


# Generates the average extra personal fouls per game the referees called on the home team versus the away team.
# A negative number means the refs called less fouls on the home team than the away team
def generate_refs_statistics(game_officials, season_referee_stats):
    home_pf_bias = 0
    # TODO: Write a forloop that gets the first and last name from the rows and calls generate_ref_statistic
    for _, row in game_officials[['FIRST_NAME', 'LAST_NAME']].iterrows():
        home_pf_bias += generate_ref_statistic(f"{row['FIRST_NAME']} {row['LAST_NAME']}", season_referee_stats)
    return home_pf_bias


# Returns a 2 element list of the home team and away team's information.
def pull_game_data(game_id, season_referee_stats, star_player_ids):
    # 2 length list, player stats are the first element.
    print(game_id)
    players_stats = boxscoreplayertrackv3\
        .BoxScorePlayerTrackV3(game_id=game_id, timeout=game_timeout).get_data_frames()
    game_box_score = players_stats[0][
        ['teamId', 'personId', 'minutes', 'speed', 'distance', 'reboundChancesOffensive',
         'reboundChancesTotal', 'touches', 'secondaryAssists', 'freeThrowAssists', 'passes',
         'assists', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted',
         'contestedFieldGoalPercentage', 'uncontestedFieldGoalsMade',
         'uncontestedFieldGoalsAttempted', 'uncontestedFieldGoalsPercentage', 'fieldGoalPercentage',
         'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted',
         'defendedAtRimFieldGoalPercentage']]
    summary = boxscoresummaryv2\
        .BoxScoreSummaryV2(game_id=game_id, timeout=game_timeout).get_data_frames()
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
    return 0 if len(recent_games) == 0 else len(starfilter(lambda _, won: won, recent_games))/len(recent_games)


# teams ONLY have 1 coach per season, so this is a major issue to work through. Currenlty, I am assuming that there
# is only one coach per year, and calculating a team's win percentage and close win percentage
def pull_team_data(game_data: DataFrame, team_id):
    team_games = game_data[game_data['teamId'] == team_id]
    # TODO: Consolidate every initialization into one statement.
    team_games.loc[:, ['WIN_PCT', 'CLOSE_WIN_PCT', 'IS_BACK_TO_BACK', 'REST_DAYS', 'Distance', 'RECENT_WIN_PCT']] = 0
    wins = 0
    close_wins = 0
    games_played = 0
    close_games_played = 0
    current_game_date = None
    previous_game_location = team_id
    recent_games = []
    for index, row in team_games.iterrows():
        team_games.at[index, 'WIN_PCT'] = (0 if games_played == 0 else wins / games_played)
        team_games.at[index, 'CLOSE_WIN_PCT'] = (0 if close_games_played == 0 else close_wins / close_games_played)
        team_games.at[index, 'RECENT_WIN_PCT'] = recent_games_win_pct(recent_games)
        previous_game_date = current_game_date
        current_game_date = pandas.to_datetime(row['GAME_DATE_EST'], format='%Y-%m-%dT%H:%M:%S')
        recent_games = update_recent_games(recent_games, current_game_date, row['WON'])
        rest_days = (firstGameRest if previous_game_date is None else (current_game_date - previous_game_date).days - 1)
        team_games.at[index, 'IS_BACK_TO_BACK'] = (False if previous_game_date is None else rest_days == 0)
        team_games.at[index, 'REST_DAYS'] = rest_days
        team_games.at[index, 'DISTANCE'] = distance_between_teams(previous_game_location, row['GAME_LOCATION'])
        previous_game_location = row['GAME_LOCATION']
        wins += int(row['WON'])
        games_played += 1
        if row['CLOSE']:
            close_wins += int(row['WON'])
            close_games_played += 1
    return team_games.copy()


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
                    season_type_all_star='Regular Season').get_data_frames()[0].sort_values('EFF', ascending=False)
    for team_id in team_ids:
        star_players[team_id] = best_players[best_players['TEAM_ID'] == team_id].head(1)['PLAYER_ID'].values[0]
    return star_players


# Generates the regular season statistics for every game in a season.
def generate_season_stats(season: str):
    # This is only for testing to test on a smaller dataset.
    game_limit = 1
    # TODO use season long stats to determine the star player per team per year,
    #  and pass that data into pull_game_data to add that extra feature.
    referees = load_referees(season)
    season_games = leaguegamelog.LeagueGameLog(counter=10, direction="ASC", league_id=nba_id,
                                               season=season, season_type_all_star='Regular Season',
                                               sorter='DATE').get_data_frames()[0]
    all_team_ids = season_games['TEAM_ID'].unique()
    star_player_ids = generate_star_players(all_team_ids, season)
    all_game_ids = season_games['GAME_ID'].unique()
    # monstrous list comprehension because there's no builtin flatten function for python lists.
    all_game_stats_list = [team_data for game in all_game_ids
                           for team_data in pull_game_data(game, referees, star_player_ids)]
    all_game_stats = pandas.concat(all_game_stats_list, axis=0, ignore_index=True)
    all_game_stats_with_team_stats = \
        pandas.concat([pull_team_data(all_game_stats, team_id) for team_id in all_team_ids],
                      axis=0,
                      ignore_index=True)
    complete_season_stats_list = [update_game_stats(game_id, all_game_stats_with_team_stats)
                                for game_id in all_game_ids]
    complete_season_stats = pandas.concat(complete_season_stats_list, axis=0, ignore_index=True)
    # TODO: Remove all of the columns that are not longer needed anymore (e.g Date, teamIds, playerIds, etc).
    return complete_season_stats


# Generates the given season and writes the data to the corresponding csv in output_data
def write_season_stats(season):
    season_stats = generate_season_stats(season)
    # Solution from https://stackoverflow.com/questions/17383094/how-can-i-map-true-false-to-1-0-in-a-pandas-dataframe
    season_stats.loc[:, ['CLOSE', 'WON', 'IS_BACK_TO_BACK', 'STAR_PLAYER_PRESENT']] =\
        season_stats[['CLOSE', 'WON', 'IS_BACK_TO_BACK', 'STAR_PLAYER_PRESENT']].astype(int)
    season_stats.to_csv(f"output_data/{season}_data.csv", index=False)


if __name__ == '__main__':
    write_season_stats('2022-23')
    # TODO: Figure out how to get coaches on a more granular level than season, since mid season changes SHOULD be
    #  reflected if we decide to use coaches
    teams_coaches = commonteamroster.CommonTeamRoster(team_id='1610612749', season='2023').get_data_frames()
    print(teams.get_teams())
    print(teams_coaches[1][['SEASON', 'FIRST_NAME', 'LAST_NAME', 'COACH_NAME', 'IS_ASSISTANT']])
