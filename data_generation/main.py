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
from pandas import DataFrame
from geopy import distance

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

# TODO: Decide how to calculate locations. BoxScores do not track locations, they track city home games.
#  However, the 'City' field of a team isn't accurate (the Golden State Warrior's city name is 'Golden State').
#  If the city names are nonsensical, it might make the most sense to just use TeamIds instead.

# A dictionary of City names to (lat, lon) pairs
CITY_LOCATIONS = {'Atlanta': (33.7501, -84.3885), 'Boston': (42.3601, -71.0589), 'Cleveland': (41.4993, -81.6944),
                  'New Orleans': (29.9509, -90.0758), 'Chicago': (41.8781, -87.6298), 'Dallas': (32.7767, -96.7970),
                  'Denver': (39.7392, -104.9903), }


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


# Given a game box score, generates a vector for both teams of the top N players (and their stats) on this team and
# the opposing team
def generate_game_stats_per_team(game_box_score):
    team_ids = list(set(game_box_score['teamId']))
    home_team_top_players = query_top_players_on_team(game_box_score, team_ids[0])
    away_team_top_players = query_top_players_on_team(game_box_score, team_ids[1])
    home_team_players_info = DataFrame({'teamId': [team_ids[0]]})
    away_team_players_info = DataFrame({'teamId': [team_ids[1]]})
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
    return home_team_players_info, away_team_players_info


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
    return home_team_result, away_team_result


# Returns a 2 element list of the home team and away team's information.
def pull_game_data(game_id):
    # 2 length list, player stats are the first element.
    players_stats = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id).get_data_frames()
    game_box_score = players_stats[0][
        ['teamId', 'teamCity', 'personId', 'minutes', 'speed', 'distance', 'reboundChancesOffensive',
         'reboundChancesTotal', 'touches', 'secondaryAssists', 'freeThrowAssists', 'passes',
         'assists', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted',
         'contestedFieldGoalPercentage', 'uncontestedFieldGoalsMade',
         'uncontestedFieldGoalsAttempted', 'uncontestedFieldGoalsPercentage', 'fieldGoalPercentage',
         'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted',
         'defendedAtRimFieldGoalPercentage']]
    summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id).get_data_frames()
    print(players_stats[1]['TEAM_CITY'].unique())
    officials = summary[2]
    scores = summary[5]
    team_ids = list(set(game_box_score['teamId']))
    home_team_players_info, away_team_players_info = generate_game_stats_per_team(game_box_score)
    # TODO: Join on an already loaded First/Last name table of referee statistics
    # officials_lineup = pandas.DataFrame(officials[['FIRST_NAME', 'LAST_NAME', 'JERSEY_NUM']].values.flatten(), [''])
    home_team_result, away_team_result = add_scoring_statistics(home_team_players_info, away_team_players_info, scores)
    return [home_team_result, away_team_result]


# teams ONLY have 1 coach per season, so this is a major issue to work through. Currenlty, I am assuming that there
# is only one coach per year, and calculating a team's win percentage and close win percentage
def pull_team_data(game_data: DataFrame, team_id):
    team_games = game_data[game_data['teamId'] == team_id]
    team_games.loc[:, 'WIN_PCT'] = 0
    team_games.loc[:, 'CLOSE_WIN_PCT'] = 0
    team_games.loc[:, 'IS_BACK_TO_BACK'] = False
    team_games.loc[:, 'REST_DAYS'] = 0
    wins = 0
    close_wins = 0
    games_played = 0
    close_games_played = 0
    current_game_date = None
    for index, row in team_games.iterrows():
        team_games.loc[index, 'WIN_PCT'] = (0 if games_played == 0 else wins / games_played)
        team_games.loc[index, 'CLOSE_WIN_PCT'] = (0 if close_games_played == 0 else close_wins / close_games_played)
        previous_game_date = current_game_date
        current_game_date = pandas.to_datetime(row['GAME_DATE_EST'], format='%Y-%m-%dT%H:%M:%S')
        rest_days = (firstGameRest if previous_game_date is None else (current_game_date - previous_game_date).days - 1)
        team_games.loc[index, 'IS_BACK_TO_BACK'] = (False if previous_game_date is None else rest_days == 0)
        team_games.loc[index, 'REST_DAYS'] = rest_days
        wins += int(row['WON'])
        games_played += 1
        if row['CLOSE']:
            close_wins += int(row['WON'])
            close_games_played += 1
    return team_games


# Generates the regular season statistics for every game in a season.
def generate_season_stats(season: str):
    # This is only for testing to test on a smaller dataset.
    game_limit = 50
    season_games = leaguegamelog.LeagueGameLog(counter=10, direction="ASC", league_id=nba_id,
                                               season=season, season_type_all_star='Regular Season',
                                               sorter='DATE').get_data_frames()[0]
    season_games = season_games[season_games['TEAM_ID'] == 1610612744]
    # monstrous list comprehension because there's no builtin flatten function for python lists.
    all_game_stats_list = [team_data for game in season_games['GAME_ID'].unique()[:game_limit]
                           for team_data in pull_game_data(game)]
    all_game_stats = pandas.concat(all_game_stats_list, axis=0, ignore_index=True)
    all_game_stats_with_team_stats = \
        pandas.concat([pull_team_data(all_game_stats, team_id) for team_id in all_game_stats['teamId'].unique()],
                      axis=0,
                      ignore_index=True)
    # TODO: Remove all of the columns that are not longer needed anymore (e.g Date, teamIds, playerIds, etc).
    return all_game_stats_with_team_stats


if __name__ == '__main__':
    test_season = generate_season_stats('2022-23')
    print(test_season[['GAME_DATE_EST', 'REST_DAYS', 'IS_BACK_TO_BACK']])
    # TODO: Figure out how to get coaches on a more granular level than season, since mid season changes SHOULD be
    #  reflected if we decide to use coaches
    teams_coaches = commonteamroster.CommonTeamRoster(team_id='1610612749', season='2023').get_data_frames()
    print(teams.get_teams())
    print(teams_coaches[1][['SEASON', 'FIRST_NAME', 'LAST_NAME', 'COACH_NAME', 'IS_ASSISTANT']])
