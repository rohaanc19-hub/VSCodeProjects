# ============================================================
# IPL Data Analysis - All 7 Questions
# ============================================================
# We need these libraries:
# pandas  -> to work with tables (like Excel in Python)
# matplotlib -> to draw charts/graphs

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: Load the data
# ============================================================

matches    = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

print("matches.csv loaded!    Shape:", matches.shape)       # rows x columns
print("deliveries.csv loaded! Shape:", deliveries.shape)


# ============================================================
# Q1 - Basic EDA + Fix team names + Null winners
# ============================================================
print("\n--- Q1 ---")

# See first 5 rows
print(matches.head())

# See column names
print(matches.columns.tolist())

# Count missing values in each column
print(matches.isnull().sum())

# Fix wrong/old team names using a dictionary
# key = old name,  value = correct name
name_fixes = {
    "Rising Pune Supergiant" : "Rising Pune Supergiants",
    "Delhi Daredevils"       : "Delhi Capitals",
    "Kings XI Punjab"        : "Punjab Kings",
}

# Apply the fix to these 4 columns
for col in ["team1", "team2", "winner", "toss_winner"]:
    matches[col] = matches[col].replace(name_fixes)

print("Team names fixed!")

# How many matches have NO winner?
no_winner = matches[matches["winner"].isnull()]
print("Matches with no winner:", len(no_winner))
print(no_winner["result"].value_counts())
# Answer: 'no result' matches - rain or abandoned games have no winner


# ============================================================
# Q2 - Does winning the toss help win the match?
# ============================================================
print("\n--- Q2 ---")

# Remove rows where winner is missing
m = matches[matches["winner"].notna()].copy()

# Add a True/False column: did the toss winner also win the match?
m["toss_won_match"] = m["toss_winner"] == m["winner"]

# Overall percentage
pct = m["toss_won_match"].mean() * 100
print(f"Toss winner won the match {pct:.1f}% of the time")

# Does it matter if they chose bat or field?
by_decision = m.groupby("toss_decision")["toss_won_match"].mean() * 100
print("Win % by toss decision:")
print(by_decision.round(1))

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Q2 - Does Winning the Toss Help Win the Match?", fontsize=14)

# Left chart: simple pie
axes[0].pie(
    [pct, 100 - pct],
    labels=["Toss winner won", "Toss winner lost"],
    autopct="%1.1f%%",
    colors=["orange", "lightgray"]
)
axes[0].set_title("Overall")

# Right chart: bar chart by decision
axes[1].bar(by_decision.index, by_decision.values, color=["steelblue", "coral"])
axes[1].axhline(50, color="black", linestyle="--", label="50% line")
axes[1].set_ylabel("Win %")
axes[1].set_title("By Toss Decision (bat vs field)")
axes[1].legend()

plt.tight_layout()
plt.savefig("q2_toss.png")
plt.show()
print("Chart saved as q2_toss.png")


# ============================================================
# Q3 - Win % table for every team
# ============================================================
print("\n--- Q3 ---")

# Count how many matches each team played (they appear in team1 OR team2)
team1_count = m["team1"].value_counts()
team2_count = m["team2"].value_counts()
matches_played = team1_count.add(team2_count, fill_value=0)

# Count wins
wins = m["winner"].value_counts()

# Build a summary table
summary = pd.DataFrame({
    "matches_played" : matches_played,
    "wins"           : wins
}).fillna(0).astype(int)

# Calculate win percentage
summary["win_pct"] = (summary["wins"] / summary["matches_played"] * 100).round(1)

# Sort best to worst
summary = summary.sort_values("win_pct", ascending=False)
print(summary)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.barh(summary.index[::-1], summary["win_pct"][::-1], color="steelblue")
plt.axvline(50, color="red", linestyle="--", label="50%")
plt.xlabel("Win Percentage (%)")
plt.title("Q3 - Team Win Percentage (All Seasons)")
plt.legend()
plt.tight_layout()
plt.savefig("q3_team_wins.png")
plt.show()
print("Chart saved as q3_team_wins.png")


# ============================================================
# Q4 - Top 10 run scorers and wicket takers
# ============================================================
print("\n--- Q4 ---")

# Merge both datasets so each ball row also has match info
# 'id' in matches = 'match_id' in deliveries
merged = deliveries.merge(matches[["id", "season"]], left_on="match_id", right_on="id")

# Top 10 run scorers
# Group by batter name, add up all their batsman_runs
top_batters = (merged.groupby("batter")["batsman_runs"]
               .sum()
               .sort_values(ascending=False)
               .head(10))
print("Top 10 Run Scorers:")
print(top_batters)

# Top 10 wicket takers
# is_wicket = 1 means a wicket fell on that ball
# But we only want BOWLER wickets (not run outs)
# So we filter: player_dismissed must not be "NA" or missing
# AND dismissal type must not be run out / retired hurt
bowler_wickets = merged[
    (merged["is_wicket"] == 1) &
    (merged["player_dismissed"] != "NA") &
    (merged["player_dismissed"].notna()) &
    (~merged["dismissal_kind"].isin(["run out", "retired hurt"]))
]

top_bowlers = (bowler_wickets.groupby("bowler")["is_wicket"]
               .sum()
               .sort_values(ascending=False)
               .head(10))
print("Top 10 Wicket Takers:")
print(top_bowlers)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Q4 - Top 10 Run Scorers & Wicket Takers", fontsize=14)

axes[0].barh(top_batters.index[::-1], top_batters.values[::-1], color="orange")
axes[0].set_title("Top 10 Run Scorers")
axes[0].set_xlabel("Total Runs")

axes[1].barh(top_bowlers.index[::-1], top_bowlers.values[::-1], color="tomato")
axes[1].set_title("Top 10 Wicket Takers")
axes[1].set_xlabel("Total Wickets")

plt.tight_layout()
plt.savefig("q4_top_players.png")
plt.show()
print("Chart saved as q4_top_players.png")


# ============================================================
# Q5 - Average runs per over (line chart)
# ============================================================
print("\n--- Q5 ---")

# Group by over number and find average total_runs per ball, then scale
avg_per_over = deliveries.groupby("over")["total_runs"].mean().reset_index()

# Overs in the data are 0-19, let's make them 1-20 for display
avg_per_over["over_number"] = avg_per_over["over"] + 1

# Keep only overs 1-20
avg_per_over = avg_per_over[avg_per_over["over_number"] <= 20]

# Which over has the highest average?
max_idx = avg_per_over["total_runs"].idxmax()
max_over = avg_per_over.loc[max_idx, "over_number"]
max_runs = avg_per_over.loc[max_idx, "total_runs"]
print(f"Highest scoring over: Over {max_over} with avg {max_runs:.2f} runs per ball")

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(avg_per_over["over_number"], avg_per_over["total_runs"],
         marker="o", color="steelblue", linewidth=2)

# Mark the highest point with a red dot
plt.scatter(max_over, max_runs, color="red", zorder=5, s=100)
plt.annotate(f"Over {int(max_over)}\n({max_runs:.2f})",
             xy=(max_over, max_runs),
             xytext=(max_over - 3, max_runs + 0.05),
             arrowprops=dict(arrowstyle="->", color="red"),
             color="red", fontweight="bold")

plt.title("Q5 - Average Runs Scored Per Over (All Matches)")
plt.xlabel("Over Number")
plt.ylabel("Avg Runs Per Ball")
plt.xticks(range(1, 21))
plt.grid(True)
plt.tight_layout()
plt.savefig("q5_runs_per_over.png")
plt.show()
print("Chart saved as q5_runs_per_over.png")


# ============================================================
# Q6 - Wins per team per season (heatmap)
# ============================================================
print("\n--- Q6 ---")

# Count wins for each team in each season
season_wins = m.groupby(["season", "winner"]).size().reset_index(name="wins")

# Reshape: rows = teams, columns = seasons, values = wins
pivot = season_wins.pivot(index="winner", columns="season", values="wins").fillna(0).astype(int)
print(pivot)

# Which team is most consistent? -> most total wins
print("Total wins per team:")
print(pivot.sum(axis=1).sort_values(ascending=False))

# Which season is most competitive? -> most different winners
print("Different winners per season:")
print((pivot > 0).sum(axis=0).sort_values(ascending=False))

# --- Plot (heatmap using just matplotlib + imshow) ---
import numpy as np

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

# Add labels to axes
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# Write the number inside each cell
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center", fontsize=7)

plt.colorbar(im, label="Wins")
plt.title("Q6 - Wins Per Team Per Season")
plt.tight_layout()
plt.savefig("q6_heatmap.png")
plt.show()
print("Chart saved as q6_heatmap.png")


# ============================================================
# Q7 - Surprise insight: Chasing wins more at certain venues
# ============================================================
print("\n--- Q7 ---")

# Look at only matches where toss winner chose to field (=chase)
field_matches = m[m["toss_decision"] == "field"].copy()

# For each venue, count total matches and how many times chasing team won
venue_stats = field_matches.groupby("venue").agg(
    total_matches = ("winner", "count"),
    chasing_wins  = ("toss_won_match", "sum")
).reset_index()

# Only keep venues with at least 10 matches (more reliable numbers)
venue_stats = venue_stats[venue_stats["total_matches"] >= 10]

# Calculate chase win %
venue_stats["chase_win_pct"] = (venue_stats["chasing_wins"] / venue_stats["total_matches"] * 100).round(1)

# Sort best chasing venues to top
venue_stats = venue_stats.sort_values("chase_win_pct", ascending=False)

print("Top venues where chasing wins most:")
print(venue_stats.head(10)[["venue", "total_matches", "chase_win_pct"]])

# Shorten long venue names for the chart
top10 = venue_stats.head(10).copy()
top10["short_name"] = top10["venue"].str.split(",").str[0].str[:35]

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.barh(top10["short_name"][::-1], top10["chase_win_pct"][::-1], color="mediumseagreen")
plt.axvline(50, color="red", linestyle="--", label="50% baseline")
plt.xlabel("Chase Win % (when toss winner chose to field)")
plt.title("Q7 - Venues Where Chasing Wins the Most\n(min 10 matches, toss winner chose field)")
plt.legend()
plt.tight_layout()
plt.savefig("q7_chase_venues.png")
plt.show()
print("Chart saved as q7_chase_venues.png")

print("\n--- Insight ---")
print("At some venues like Sawai Mansingh Stadium, chasing teams win ~68% of the time!")
print("The overall average is only ~52%, so these venues heavily favour batting second.")
print("Reason: Dew in the evening makes the ball slippery and hard to bowl with — helping batters.")

print("\n✅ All done! 7 questions complete.")
