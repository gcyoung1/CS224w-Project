import sqlite3
import pickle
import os
sqlGetTable = "SELECT * FROM %s" # Select all from a table
dbFileName = 'database.sqlite'
dbSaveFileName = 'soccer_%s.txt'
pickleFileName = 'soccer.pkl'


class Match:
    def __init__(self, row):
        """
        Stuff not Added
        id;	goal;	shoton;	shotoff;	foulcommit;	card;	cross;	corner;
        possession;	B365H;	B365D;	B365A;	BWH;	BWD;	BWA;	IWH;
            IWD;	IWA;	LBH;	LBD;	LBA;	PSH;	PSD;	PSA;
            WHH;	WHD;	WHA;	SJH;	SJD;	SJA;	VCH;	VCD;
            VCA;	GBH;	GBD;	GBA;	BSH;	BSD;	BSA
        """
        self.id = row['match_api_id']
        # Location
        self.countryId = row['country_id']
        # Results
        self.home_goal = row['home_team_goal']
        self.away_goal = row['away_team_goal']
        # Teams
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
        # Time
        self.leagueId = row['league_id']
        self.season = row['season']
        self.stageId = row['stage']
        # Misc
        self.date = row['date'] # I think this is the date of the datapoint creation

class MetaData:
    def __init__(self):
        # country ID -> country name
        self.country = {}
        # team ID -> {'name': string team name, 'country' : home country ID}
        self.team = {}
        # player ID -> {'name': string name, 'team': list of teamID, 'bday': birthday, 'h': height, 'w': weight}
        self.player = {}

    def AddCountry(self, c_row):
        self.country[c_row['id']] = c_row['name']

    def AddMatchInfo(self, match):
        htId, cId = match.home_team, match.countryId
        if htId is None:
            print("Match %d has no home team?" % match.id)
            return
        if htId not in self.team:
            self.team[htId] = {'name': '', 'country': cId}
            return
        ht = self.team[htId]
        if cId is not None:
            if ht['country'] not in (-1, cId):
                print("Team/Country error: Team %d at home in country %d and %d" % (htId, ht['country'], cId))
            ht['country'] = cId
        for pl in match.home_players:
            if pl is None:
                continue
            if pl not in self.player:
                print("Player %d not in players table" % pl)
                continue
            if htId not in self.player[pl]['team']:
                self.player[pl]['team'].append(htId)


    def AddPlayer(self, p_row):
        self.player[p_row['player_api_id']] = {
            'name': p_row['player_name'],
            'team': [],
            'bday': p_row['birthday'],
            'h': p_row['height'],
            'w': p_row['weight'],
        }

    def AddTeam(self, t_row):
        name = t_row['team_long_name']
        id = t_row['team_api_id']
        if id not in self.team:
            self.team[id] = {'name': name, 'country': -1}
        else:
            self.team[id]['name'] = name

""" 
    Initial exploration, figure out what tables there are and -> dbSaveFileName
    Other stuff I learned: "stage" is how far into a league a match is
        each team plays 38 games in a league, so "stage" in [1, 38] is the number
        of that game
"""
def getTablesInfo():
    connection = sqlite3.connect(dbFileName)
    connection.row_factory = sqlite3.Row
    c = connection.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    ret = c.fetchall()
    tableNames = [r['name'] for r in ret]
    lines = []
    for tableName in tableNames:
        c.execute(sqlGetTable % tableName)
        ret = c.fetchall()
        keys = ret[0].keys()
        lines.append("Table: %s\n" % tableName)
        lines.append("\tTotal entries %d\n" % len(ret))
        lines.append("\tNum columns %s\n" % len(keys))
        lines.append("\t%s\n" % ";\t".join(keys))
    with open(dbSaveFileName % "tables", 'a+') as f:
        f.writelines(lines)

# create a MetaData obj and list of matches from the SQLite file, save to pickle
def exploreSQL(saveToPickle = True):
    connection = sqlite3.connect(dbFileName)
    connection.row_factory = sqlite3.Row
    c = connection.cursor()

    meta = MetaData()
    c.execute(sqlGetTable % "Team")
    [meta.AddTeam(row) for row in c.fetchall()]
    
    c.execute(sqlGetTable % "Player")
    [meta.AddPlayer(row) for row in c.fetchall()]
    
    c.execute(sqlGetTable % "Country")
    [meta.AddCountry(row) for row in c.fetchall()]

    c.execute(sqlGetTable % "Match")
    matches = [Match(r) for r in c.fetchall()]
    [meta.AddMatchInfo(match) for match in matches]

    if saveToPickle:
        with open(pickleFileName, 'wb+') as f:
            pickle.dump([meta, matches], f)

    return meta, matches

# load the MetaData obj and list of matches from the pickle file
def loadPickle():
    if not os.path.exists(pickleFileName):
        print("Error: %s does not exist in directory" % pickleFileName)
        return None
    with open(pickleFileName, 'rb') as f:
        return pickle.load(f)







