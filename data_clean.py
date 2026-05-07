import polars as pl
from pathlib import Path


data_dir = Path("./data")

team_renames = {"Utah Hockey Club": "Utah Mammoth"}
df = (
    pl.read_csv(
        data_dir / "games.csv",
    )
    .with_columns(
        (pl.col("Home_G") - pl.col("Away_G")).alias("Goal_Diff"),
        pl.concat_str(pl.col("Date"), pl.col("Time"), separator=" ")
        .str.strptime(pl.Datetime, "%Y-%m-%d %I:%M %p")
        .alias("Datetime"),
        pl.col("Home").replace(team_renames).alias("Home"),
        pl.col("Away").replace(team_renames).alias("Away"),
        pl.when(pl.col("OT").is_null()).then(0).otherwise(1).alias("OT"),
    )
    .drop(["Date", "Time"])
)

teams = set(df["Home"].unique())

df = (
    df.with_columns(
        pl.when(pl.col("Home") == team)
        .then(1)
        .when(pl.col("Away") == team)
        .then(-1)
        .otherwise(0)
        .alias(team)
        for team in teams
    )
    .with_columns(
        pl.when(pl.col("Goal_Diff") > 0)
        .then(1)
        .when(pl.col("Goal_Diff") < 0)
        .then(0)
        .alias("home_win")
    )
    .drop(["Home", "Away"])
)

df.write_csv(data_dir / "cleaned_games.csv")
with open(data_dir / "teams.txt", "w") as f:
    for team in teams:
        f.write(team + "\n")
