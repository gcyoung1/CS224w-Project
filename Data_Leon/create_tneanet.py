import sqlite_process as sp
import snap
TN = snap.TNEANet
saveFileName = "soccer.graph"


def load_tneanet():
    return TN.Load(snap.TFIn(saveFileName))

def create_tneanet(save = True):
    meta, matches = sp.loadPickle()
    numNodes = len(matches) + len(meta.player) + len(meta.team) + len(meta.country)
    numEdges = 3 * len(matches) + 4 * len(meta.player) + len(meta.team)
    G = TN.New(numNodes, numEdges)
    countryToNId = {}
    teamToNId = {}
    playerToNId = {}
    matchIndToNId = []
    i = 0
    for (countryId, countryName) in meta.country.items():
        ni = G.GetNI(G.AddNode(i))
        G.AddStrAttrDatN(ni, "country", "kind")
        G.AddIntAttrDatN(ni, countryId, "countryId")
        G.AddStrAttrDatN(ni, countryName, "countryName")
        countryToNId[countryId] = i
        i += 1
    for (teamId, d) in meta.team.items():
        ni = G.GetNI(G.AddNode(i))
        G.AddStrAttrDatN(ni, "team", "kind")
        G.AddIntAttrDatN(ni, teamId, "teamId")
        G.AddStrAttrDatN(ni, d['name'], "teamName")
        teamToNId[teamId] = i
        EId = G.AddEdge(i, countryToNId[d['country']])
        G.AddStrAttrDatE(EId, "team from", "kind")
        i += 1
    for (playerId, d) in meta.player.items():
        ni = G.GetNI(G.AddNode(i))
        G.AddStrAttrDatN(ni, "player", "kind")
        G.AddIntAttrDatN(ni, playerId, "playerId")
        G.AddStrAttrDatN(ni, d['name'], "playerName")
        playerToNId[playerId] = i
        for teamId in d['team']:
            EId = G.AddEdge(i, teamToNId[teamId])
            G.AddStrAttrDatE(EId, "plays for", "kind")
        i += 1
    for match in matches:
        matchIndToNId.append(i)
        ni = G.GetNI(G.AddNode(i))
        G.AddStrAttrDatN(ni, "match", "kind")
        G.AddIntAttrDatN(ni, match.away_goal, "away_goal")
        G.AddIntAttrDatN(ni, match.home_goal, "away_goal")
        G.AddIntAttrDatN(ni, match.stageId, "stageId")
        G.AddStrAttrDatN(ni, match.season, "season")
        G.AddIntAttrDatN(ni, match.leagueId, "leagueId")
        G.AddIntAttrDatN(ni, match.id, "matchId")
        EId = G.AddEdge(i, teamToNId[match.home_team])
        G.AddStrAttrDatE(EId, "home team", "kind")
        EId = G.AddEdge(i, teamToNId[match.away_team])
        G.AddStrAttrDatE(EId, "away team", "kind")
        EId = G.AddEdge(i, countryToNId[match.countryId])
        G.AddStrAttrDatE(EId, "match in", "kind")
        i+=1
    if save:
        G.Save(snap.TFOut(saveFileName))
    return G




