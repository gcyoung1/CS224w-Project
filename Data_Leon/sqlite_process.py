import sqlite3
import pickle
import os
import numpy as np
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

    c.execute(sqlGetTable % "Player_Attributes")
    [row for row in c.fetchall()]
    
    c.execute(sqlGetTable % "Country")
    [meta.AddCountry(row) for row in c.fetchall()]

    c.execute(sqlGetTable % "Match")
    matches = [Match(r) for r in c.fetchall()]
    [meta.AddMatchInfo(match) for match in matches]

    if saveToPickle:
        with open(pickleFileName, 'wb+') as f:
            pickle.dump([meta, matches], f)

    return meta, matches

from collections import defaultdict
class PlayerAttributes:
    def __init__(self):
        pref_foot_dict = {'right': 1, 'left': -1, None: 0}
        Atk_wr_dict = {'low': 1, 'medium': 2, 'high': 3}
        Def_wr_dict = {'_0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                       '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                       'low': 2, 'medium': 5, 'high': 8}
        attrs = ["overall_rating", "potential", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
                 "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration",
                 "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength",
                 "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking",
                 "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning",
                 "gk_reflexes"]
        def deNone(v):
            return 0 if v is None else v
        connection = sqlite3.connect(dbFileName)
        connection.row_factory = sqlite3.Row
        c = connection.cursor()
        c.execute(sqlGetTable % "Player_Attributes")
        pid_attr = defaultdict(list)
        for row in c.fetchall():
            id = row['player_api_id']
            date = self.dateTuple(row['date'])
            pf = row["preferred_foot"]
            pf = pref_foot_dict[pf] if pf in pref_foot_dict else -1
            atk = row["attacking_work_rate"]
            atk = Atk_wr_dict[atk] if atk in Atk_wr_dict else -1
            df = row["defensive_work_rate"]
            df = Def_wr_dict[df] if df in Def_wr_dict else -1
            other_attrs = [deNone(row[attr]) for attr in attrs]
            cur_attrs = (date, np.array([pf, atk, df, *other_attrs]))
            pid_attr[id].append(cur_attrs)
        self.player_attrs = pid_attr
    @staticmethod
    def dateTuple(s):
        return (int(s[:4]), int(s[5:7]), int(s[8:10]))
    def getPlayerAttr(self, pid, time):
        if type(time) != tuple:
            time = self.dateTuple(time)
        if pid not in self.player_attrs:
            return np.zeros(38)
        attrs = self.player_attrs[pid]
        times = [attr[0] for attr in attrs]
        ind = len(times)-1
        while ind > 0 and times[ind] < time:
            ind -= 1
        ind = min(ind+1, len(times)-1)
        return attrs[ind][1]


predictor_dset_names = ['attrsToMatchResults.npz', 'attrsToMatchResults2.npz',
                        'embedToMatchResults.npz', 'embedToMatchResults2.npz',
                        'attrsAndEmbedsToMatchResults.npz', 'attrsAndEmbedsToMatchResults.npz']
def make_predictor_dset(two_cat=False, embeddings=None, G=None, only_embeddings=False):
    def get_id_to_nids(G):
        if G is None:
            return None, None, None, None
        pid_to_nid = {}
        cid_to_nid = {}
        tid_to_nid = {}
        mid_to_nid = {}
        for ni in G.Nodes():
            kind = G.GetStrAttrDatN(ni, "kind")
            if kind == 'player':
                pid_to_nid[G.GetIntAttrDatN(ni, "playerId")] = ni.GetId()
            elif kind == 'country':
                cid_to_nid[G.GetIntAttrDatN(ni, "countryId")] = ni.GetId()
            elif kind == 'team':
                tid_to_nid[G.GetIntAttrDatN(ni, "teamId")] = ni.GetId()
            elif kind == 'match':
                mid_to_nid[G.GetIntAttrDatN(ni, "matchId")] = ni.GetId()
        return pid_to_nid, cid_to_nid, tid_to_nid, mid_to_nid
    _, matches = loadPickle()
    # filter all matches that don't have all player info
    matches = [m for m in matches if None not in m.home_players and None not in m.away_players]
    results = np.zeros(len(matches), dtype=np.int64)
    feats = np.zeros((len(matches), 836 * (not only_embeddings) + (26*embeddings[0].shape[0] if embeddings else 0)), dtype=np.float32)
    pa = PlayerAttributes()
    pid_to_nid, cid_to_nid, tid_to_nid, mid_to_nid = get_id_to_nids(G)
    for i, match in enumerate(matches):
        date = match.date
        if embeddings:
            if only_embeddings:
                feats[i] = np.concatenate([embeddings[pid_to_nid[p]] for p in (match.home_players+match.away_players)] +\
                                      [embeddings[tid_to_nid[match.home_team]], embeddings[tid_to_nid[match.away_team]],
                                        embeddings[cid_to_nid[match.countryId]], embeddings[mid_to_nid[match.id]]])
            else:
                feats[i] = np.concatenate([pa.getPlayerAttr(p, date) for p in (match.home_players+match.away_players)] +
                                      [embeddings[pid_to_nid[p]] for p in (match.home_players+match.away_players)] +\
                                      [embeddings[tid_to_nid[match.home_team]], embeddings[tid_to_nid[match.away_team]],
                                        embeddings[cid_to_nid[match.countryId]], embeddings[mid_to_nid[match.id]]])
        else:
            feats[i] = np.concatenate([pa.getPlayerAttr(p, date) for p in (match.home_players+match.away_players)])
        if two_cat:
            # 0 for home win, 1 for away win
            results[i] = 0 if match.home_goal > match.away_goal else 1
        else:
            # 0 for home win, 1 for away win, 2 for tie
            results[i] = 0 if match.home_goal > match.away_goal else 1 if match.home_goal < match.away_goal else 2
    f_ind = (0 if (not embeddings) else 2 if only_embeddings else 4) + 1 * two_cat
    np.savez(predictor_dset_names[f_ind], feats, results)
    return

def load_matchDS(two_cat=False, embeddings=False, only_embeddings=False):
    f_ind = (0 if (not embeddings) else 2 if only_embeddings else 4) + 1 * two_cat
    f = np.load(predictor_dset_names[f_ind])
    return f['arr_0'], f['arr_1']


def findPlayerAttrRanges(c):
    """
    Results::
    Prefered foot: {'right': 138409, 'left': 44733, None: 836}
    Atk_wr: {'medium': 125070, 'high': 42823, None: 3230, 'low': 8569, 'None': 3639, 'le': 104, 'norm': 348, 'stoc': 89, 'y': 106})
    Def_wr: {'medium': 130846, 'high': 27041, 'low': 18432, '_0': 2394, None: 836, '5': 234, 'ean': 104, 'o': 1550,
            '1': 441, 'ormal': 348, '7': 217, '2': 342, '8': 78, '4': 116, 'tocky': 89, '0': 197, '3': 258,
            '6': 197, '9': 152, 'es': 106})
    Min_date: (2007, 2, 22)
    Max_date:
    Min_stats: All zeros b/c of Nones
    Max_stats: array([94., 97., 95., 97., 98., 97., 93., 97., 94., 97., 97., 97., 97.,
       97., 96., 96., 96., 97., 96., 96., 96., 96., 97., 96., 96., 97.,
       96., 96., 95., 95., 94., 93., 97., 96., 96.])
    :param c:
    :return:
    """
    def dateTuple(s):
        return (int(s[:4]), int(s[5:7]), int(s[8:10]))
    def AttrRowToVec(row):
        def deNone(v):
            return 0 if v is None else v
        val = [row["overall_rating"], row["potential"], row["crossing"], row["finishing"], row["heading_accuracy"],
             row["short_passing"], row["volleys"], row["dribbling"], row["curve"], row["free_kick_accuracy"],
             row["long_passing"], row["ball_control"], row["acceleration"], row["sprint_speed"], row["agility"],
             row["reactions"], row["balance"], row["shot_power"], row["jumping"], row["stamina"], row["strength"],
             row["long_shots"], row["aggression"], row["interceptions"], row["positioning"], row["vision"],
             row["penalties"], row["marking"], row["standing_tackle"], row["sliding_tackle"], row["gk_diving"],
             row["gk_handling"], row["gk_kicking"], row["gk_positioning"], row["gk_reflexes"]]
        return np.array([deNone(v) for v in val])
    if c is None:
        connection = sqlite3.connect(dbFileName)
        connection.row_factory = sqlite3.Row
        c = connection.cursor()
    c.execute(sqlGetTable % "Player_Attributes")
    prefered_foot, atk_wr, def_wr = defaultdict(int), defaultdict(int), defaultdict(int)
    min_date = (3000, 0, 0)
    max_date = (0, 0, 0)
    min_stats = np.ones(35) * 10000
    max_stats = np.zeros(35)
    i = 0
    for row in c.fetchall():
        if i%1024 ==0:
            print(i, end=',\t')
            if i%8192==0:
                print()
        i+=1
        prefered_foot[row["preferred_foot"]] += 1
        atk_wr[row["attacking_work_rate"]] += 1
        def_wr[row["defensive_work_rate"]] += 1
        date = dateTuple(row["date"])
        min_date = min(date, min_date)
        max_date = max(date, max_date)
        stats = AttrRowToVec(row)
        min_stats = np.min(np.stack([stats, min_stats], axis=1), axis=1)
        max_stats = np.max(np.stack([stats, max_stats], axis=1), axis=1)
    return prefered_foot, atk_wr, def_wr, min_date, max_date, min_stats, max_stats

# load the MetaData obj and list of matches from the pickle file
def loadPickle():
    if not os.path.exists(pickleFileName):
        print("Error: %s does not exist in directory" % pickleFileName)
        return None
    with open(pickleFileName, 'rb') as f:
        return pickle.load(f)







