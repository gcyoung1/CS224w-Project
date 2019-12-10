def getMatchInfo(match):
  ht = meta.team[match.home_team]
  at = meta.team[match.away_team]
  hp = [match.home_players[0], match.home_players[3], match.home_players[6]]
  hpnames = [meta.player[i]['name'] if i is not None else "None" for i in hp]
  ap = [match.away_players[0], match.away_players[3], match.away_players[6]]
  apnames = [meta.player[i]['name'] if i is not None else "None" for i in ap]
  print("Home team: %s from %s;\n\t Players %s %s %s" % (ht['name'], meta.country[ht['country']], *hpnames))
  print("Away team: %s from %s;\n\t Players %s %s %s" % (at['name'], meta.country[at['country']], *apnames))
  print("Home team %d, country %d;\n\t Players %s %s %s" % (match.home_team, ht['country'], *hp))
  print("Home team %d, country %d;\n\t Players %s %s %s" % (match.away_team, at['country'], *ap))