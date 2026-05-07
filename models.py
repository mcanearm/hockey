from numpyro import distributions as dist
import numpyro
import polars as pl
import jax.numpy as jnp
from jax import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import datetime as dt
from matplotlib import pyplot as plt

import arviz as az

numpyro.set_host_device_count(4)

data_dir = Path("./data")
df = (
    pl.read_csv(data_dir / "cleaned_games.csv")
    .filter(
        pl.col("Datetime").str.strptime(pl.Datetime) >= dt.datetime(2025, 10, 1),
        pl.col("Arizona Coyotes") == 0,
    )
    .drop("Arizona Coyotes")
)

# current season games only

teams = (
    pl.read_csv(data_dir / "teams.txt", has_header=False)
    .filter(pl.col("column_1") != "Arizona Coyotes")
    .to_series()
    .to_list()
)

win_losses = df[teams].to_jax()
home_wins = df["home_win"].to_jax()
goal_diffs = df["Goal_Diff"].to_jax()

train_x, test_x, train_y, test_y = train_test_split(
    win_losses, home_wins, test_size=0.2, random_state=42
)


def model(x, y=None):
    coeffs = numpyro.sample(
        "coeffs", dist.Normal(jnp.zeros(len(teams)), jnp.ones(len(teams)))
    )
    # home team advantage
    intercept = numpyro.sample("intercept", dist.Normal(0, 1))
    logits = jnp.dot(x, coeffs) + intercept

    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


kernel = numpyro.infer.NUTS(model)
key = random.key(19900330)

mcmc = numpyro.infer.MCMC(kernel, num_warmup=2000, num_samples=5000, num_chains=4)
mcmc.run(key, train_x, train_y)

trace = az.from_numpyro(mcmc)

mcmc.print_summary()

_ = az.plot_trace(trace)
plt.show()
