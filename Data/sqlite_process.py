import sqlite3
dbfile = 'database.sqlite'
dbSaveFileName = 'soccer_%s.txt'

commands = [
    "SELECT name FROM sqlite_master WHERE type='table';",  # list of table names, in 1-tuples
    "SELECT * FROM %s",  # Select all from a table
]

class Match:
    """
    Stuff not Added
    id;	league_id;	season;	date;
    ;goal;	shoton;	shotoff;	foulcommit;	card;	cross;	corner;
    possession;	B365H;	B365D;	B365A;	BWH;	BWD;	BWA;	IWH;
        IWD;	IWA;	LBH;	LBD;	LBA;	PSH;	PSD;	PSA;
        WHH;	WHD;	WHA;	SJH;	SJD;	SJA;	VCH;	VCD;
        VCA;	GBH;	GBD;	GBA;	BSH;	BSD;	BSA
    """
    def __init__(self, row):
        self.id = row['match_api_id']
        # location
        self.countryId = row['country_id']
        self.stageId = row['stage']
        # results
        self.home_goal = row['home_team_goal']
        self.away_goal = row['away_team_goal']
        # teams
        self.home_team = row['home_team_api_id']
        self.away_team = row['away_team_api_id']
        self.home_players = [row['home_player_1'], row['home_player_2'],
             row['home_player_3'], row['home_player_4'], row['home_player_5'],
             row['home_player_6'], row['home_player_7'], row['home_player_8'],
             row['home_player_9'], row['home_player_10'], row['home_player_11']]
        self.away_players = [row['away_player_1'], row['away_player_2'],
             row['away_player_3'], row['away_player_4'], row['away_player_5'],
             row['away_player_6'], row['away_player_7'], row['away_player_8'],
             row['away_player_9'], row['away_player_10'], row['away_player_11']]

class MetaData:
    def __init__(self):
        # country ID -> country name
        self.country = {}
        # stage ID -> country ID
        self.stage = {}
        # team ID -> {'name': string team name, 'stage': home stage ID, 'country' : home country ID}
        self.team = {}
        # player ID -> {'name': string team name, 'bday': birthday, 'h': height, 'w': weight}
        self.player = {}

    def AddCountry(self, c_row):
        self.country[c_row['id']] = c_row['name']

    def _AddStage(self, match):
        if match.stageId is None or match.countryId is None:
            return
        if match.stageId in self.stage and self.stage[match.stageId] != match.countryId:
            print("Stage error %d, in country %d and %d" % (match.stageId, self.stage[match.stageId], match.countryId))
        else:
            self.stage[match.stageId] = match.countryId

    def _AddTeambyMatch(self, match):
        # add checks for conflicts
        if match.home_team is not None:
            if match.home_team not in self.team:
                self.team[match.home_team] = {}
            if match.stageId is not None:
                self.team[match.home_team]['stage'] = match.stageId
            if match.countryId is not None:
                self.team[match.home_team]['country'] = match.countryId

    def AddMatchInfo(self, match):
        self._AddStage(match)
        self._AddTeambyMatch(match)

    def AddPlayer(self, p_row):
        self.player[p_row['player_api_id']] = {
            'name': p_row['player_name'],
            'bday': p_row['birthday'],
            'h': p_row['height'],
            'w': p_row['weight'],
        }

    def AddTeam(self, t_row):
        name = t_row['team_long_name']
        id = t_row['team_api_id']
        if id not in self.team:
            self.team[id] = {'name': name, 'stage': -1, 'country': -1}
        else:
            self.team[id]['name'] = name




def getTablesInfo():
    connection = sqlite3.connect(dbfile)
    connection.row_factory = sqlite3.Row
    c = connection.cursor()
    c.execute(commands[0])
    ret = c.fetchall()
    tableNames = [r['name'] for r in ret]
    lines = []
    for tableName in tableNames:
        c.execute(commands[1] % tableName)
        ret = c.fetchall()
        keys = ret[0].keys()
        lines.append("Table: %s\n" % tableName)
        lines.append("\tTotal entries %d\n" % len(ret))
        lines.append("\tNum columns %s\n" % len(keys))
        lines.append("\t%s\n" % ";\t".join(keys))
    with open(dbSaveFileName % "tables", 'a+') as f:
        f.writelines(lines)

def main():
    connection = sqlite3.connect(dbfile)
    connection.row_factory = sqlite3.Row
    c = connection.cursor()

    # get matches
    c.execute(commands[1] % "Match")
    ret = c.fetchall()



