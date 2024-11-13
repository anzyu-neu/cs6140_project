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


# Given a game box score, generates a vector for both teams of the top N players (and their stats) on this team and
# the opposing team
def generate_game_stats_per_team(game_box_score):
    # Filter players to get top players for home and away teams based on teamId
    home_team_top_players = query_top_players_on_team(game_box_score, game_box_score.iloc[0]['teamId'])
    away_team_top_players = query_top_players_on_team(game_box_score, game_box_score.iloc[-1]['teamId'])

    # Get team IDs directly from these DataFrames
    home_team_id = home_team_top_players.iloc[0]['teamId']
    away_team_id = away_team_top_players.iloc[0]['teamId']

    # Collect player data for each team without adding teamId in individual dictionaries
    home_team_data = []
    away_team_data = []

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


# Adds the scoring, W/L, and closeness information to each team's output vector from generate_game_stats_per_team.
# Both vectors must be passed in in order to determine which team won/lost.
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
def pull_game_data(game_id, season_referee_stats):
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
    home_location = summary[0]['HOME_TEAM_ID'].values[0]
    referees = summary[2]
    scores = summary[5]
    team_ids = list(set(game_box_score['teamId']))
    home_team_players_info, away_team_players_info = generate_game_stats_per_team(game_box_score)
    ref_home_bias = generate_refs_statistics(referees, season_referee_stats)
    home_team_players_info['REF_BIAS'] = ref_home_bias
    away_team_players_info['REF_BIAS'] = -1 * ref_home_bias
    # TODO: Join on an already loaded First/Last name table of referee statistics
    # officials_lineup = pandas.DataFrame(officials[['FIRST_NAME', 'LAST_NAME', 'JERSEY_NUM']].values.flatten(), [''])
    home_team_result, away_team_result = add_scoring_statistics(home_team_players_info, away_team_players_info, scores)
    home_team_result = add_game_location(home_team_result, home_location)
    away_team_result = add_game_location(away_team_result, home_location)
    return [home_team_result, away_team_result]


def distance_between_teams(team_id_1, team_id_2):
    location_1 = CITY_LOCATIONS[team_id_1]
    location_2 = CITY_LOCATIONS[team_id_2]
    return distance.distance(location_1, location_2).miles


# teams ONLY have 1 coach per season, so this is a major issue to work through. Currently, I am assuming that there
# is only one coach per year, and calculating a team's win percentage and close win percentage
def pull_team_data(game_data: DataFrame, team_id):
    team_games = game_data[game_data['teamId'] == team_id].copy()

    # Initialize lists to store column data 
    win_pct_list = []
    close_win_pct_list = []
    is_back_to_back_list = []
    rest_days_list = []
    distance_list = []

    # Counters/placeholders initialization
    wins = 0
    close_wins = 0
    games_played = 0
    close_games_played = 0
    previous_game_date = None  # Start with None to handle the first game separately
    previous_game_location = team_id

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

    return team_games



def load_referees(season: str):
    return read_csv(f'referee_data/{season}.csv', header=[0, 1])


# Generates the regular season statistics for every game in a season.
def generate_season_stats(season: str):
    # This is only for testing to test on a smaller dataset.
    game_limit = 10
    referees = load_referees(season)
    season_games = leaguegamelog.LeagueGameLog(counter=10, direction="ASC", league_id=nba_id,
                                               season=season, season_type_all_star='Regular Season',
                                               sorter='DATE').get_data_frames()[0]
    # monstrous list comprehension because there's no builtin flatten function for python lists.
    # list of game DFs
    all_game_stats_list = [team_data for game in season_games['GAME_ID'].unique()[:game_limit]
                           for team_data in pull_game_data(game, referees)]
    all_game_stats = pd.concat(all_game_stats_list, axis=0, ignore_index=True)

    # Build list of team data and concat
    all_teams_data = [pull_team_data(all_game_stats, team_id) for team_id in all_game_stats['teamId'].unique()]
    all_game_stats_with_team_stats = pd.concat(all_teams_data, axis=0, ignore_index=True)
    
    # TODO: Remove all of the columns that are not longer needed anymore (e.g Date, teamIds, playerIds, etc).
    return all_game_stats_with_team_stats


if __name__ == '__main__':
    test_season = generate_season_stats('2022-23')
    print(test_season[['GAME_DATE_EST', 'REST_DAYS', 'IS_BACK_TO_BACK', 'DISTANCE', 'REF_BIAS']])
    # TODO: Figure out how to get coaches on a more granular level than season, since mid season changes SHOULD be
    #  reflected if we decide to use coaches
    teams_coaches = commonteamroster.CommonTeamRoster(team_id='1610612749', season='2023').get_data_frames()
    print(teams.get_teams())
    print(teams_coaches[1][['SEASON', 'FIRST_NAME', 'LAST_NAME', 'COACH_NAME', 'IS_ASSISTANT']])
