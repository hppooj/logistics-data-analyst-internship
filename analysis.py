import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
rng = np.random.default_rng(42)
N = 5000
dates = pd.to_datetime("2026-07-01") + pd.to_timedelta(rng.integers(0, 92, N), unit="D")
df = pd.DataFrame({
    "shipment_id": np.arange(1, N+1),
    "ship_date": dates,
    "warehouse": rng.choice(["WH-A","WH-B","WH-C"], N, p=[.45,.35,.20]),
    "region": rng.choice(["North","South","East","West","Central"], N, p=[.2,.25,.2,.15,.2]),
    "carrier": rng.choice(["Carrier X","Carrier Y","Carrier Z"], N, p=[.4,.35,.25]),
    "priority": rng.choice(["Standard","Express"], N, p=[.8,.2]),
    "weather": rng.choice(["Clear","Rain","Storm"], N, p=[.7,.22,.08]),
})
reg_dist = {"North":220,"South":380,"East":300,"West":450,"Central":150}
df["distance_km"] = (df["region"].map(reg_dist) * rng.lognormal(0, .25, N)).round(1)
df["weight_kg"] = (rng.lognormal(3.2, .8, N)).round(1).clip(1, 900)
df["dow"] = df["ship_date"].dt.dayofweek
df["vehicle"] = np.where(df["weight_kg"] > 120, "Truck", np.where(df["weight_kg"] > 30, "Van", "Bike/Car"))
# Delivery time (days)
base = 0.6 + df["distance_km"]/260
carrier_eff = df["carrier"].map({"Carrier X":0, "Carrier Y":0.25, "Carrier Z":0.7})
wx = df["weather"].map({"Clear":0,"Rain":0.35,"Storm":1.2})
wk = np.where(df["dow"].isin([4,5]), 0.4, 0)           # Fri/Sat pickups wait over weekend
exp = np.where(df["priority"]=="Express", -0.5, 0)
df["delivery_days"] = (base + carrier_eff + wx + wk + exp + rng.gamma(2, .25, N)).clip(0.3).round(2)
df["planned_days"] = (0.6 + df["distance_km"]/260 + 1.9 + np.where(df["priority"]=="Express", -0.5, 0)).round(2)
df["is_late"] = (df["delivery_days"] > df["planned_days"]).astype(int)
# Cost
rate = df["carrier"].map({"Carrier X":0.55,"Carrier Y":0.50,"Carrier Z":0.62})
df["transport_cost"] = (15 + df["distance_km"]*rate + df["weight_kg"]*0.35 + np.where(df["priority"]=="Express", 25, 0)
                        + rng.normal(0, 12, N)).clip(8).round(2)
# inject a few anomalies
idx = rng.choice(N, 40, replace=False); df.loc[idx, "transport_cost"] *= 2.5
df["cost_per_kg"] = df["transport_cost"]/df["weight_kg"]
df.to_csv("logistics_sim.csv", index=False)

out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
P("SHAPE", df.shape)
P(df[["distance_km","weight_kg","delivery_days","planned_days","transport_cost","cost_per_kg"]].describe().round(2).T.to_string())
P("median/mean/mode delivery", df.delivery_days.median(), round(df.delivery_days.mean(),2), df.delivery_days.round(1).mode()[0])
P("skew", df[["delivery_days","transport_cost","weight_kg","distance_km"]].skew().round(2).to_dict())
P("OTD", round(100*(1-df.is_late.mean()),1))
P("OTD by carrier", (100*(1-df.groupby("carrier").is_late.mean())).round(1).to_dict())
P("mean days by carrier", df.groupby("carrier").delivery_days.mean().round(2).to_dict())
P("mean cost by carrier", df.groupby("carrier").transport_cost.mean().round(2).to_dict())
P("OTD by weather", (100*(1-df.groupby("weather").is_late.mean())).round(1).to_dict())
P("OTD by region", (100*(1-df.groupby("region").is_late.mean())).round(1).to_dict())
P("OTD by dow", (100*(1-df.groupby("dow").is_late.mean())).round(1).to_dict())
P("OTD by priority", (100*(1-df.groupby("priority").is_late.mean())).round(1).to_dict())
P("OTD by warehouse", (100*(1-df.groupby("warehouse").is_late.mean())).round(1).to_dict())
P("mean cost by vehicle", df.groupby("vehicle").transport_cost.mean().round(2).to_dict(), df.vehicle.value_counts().to_dict())
P("cost/kg by vehicle", df.groupby("vehicle").cost_per_kg.median().round(2).to_dict())
P("mean cost by region", df.groupby("region").transport_cost.mean().round(2).to_dict())
P("share weather", df.weather.value_counts(normalize=True).round(3).to_dict())
corr = df[["distance_km","weight_kg","delivery_days","planned_days","transport_cost","is_late"]].corr().round(2)
P(corr.to_string())
# IQR outliers in cost
q1,q3 = df.transport_cost.quantile([.25,.75]); iqr=q3-q1
P("cost IQR outliers", int(((df.transport_cost>q3+1.5*iqr)).sum()), "upper bound", round(q3+1.5*iqr,1))
# regression slope cost vs distance
sl = np.polyfit(df.distance_km, df.transport_cost, 1); P("cost~distance slope/intercept", sl.round(3))
# weekly volume
wk = df.set_index("ship_date").resample("W").agg(shipments=("shipment_id","count"), otd=("is_late", lambda s: 100*(1-s.mean())))
P(wk.round(1).to_string())
# late rate carrier x weather
pv = df.pivot_table(index="carrier", columns="weather", values="is_late", aggfunc="mean").mul(100).round(1)
P(pv.to_string())
pv2 = df.pivot_table(index="region", columns="carrier", values="is_late", aggfunc="mean").mul(100).round(1)
P(pv2.to_string())
open("stats.txt","w").write("\n".join(out))

# ---------- Charts ----------
fig, ax = plt.subplots(figsize=(7,4))
sns.histplot(df.delivery_days, bins=40, kde=True, ax=ax, color="#2E75B6")
ax.axvline(df.delivery_days.mean(), color="red", ls="--", label=f"Mean {df.delivery_days.mean():.2f}")
ax.axvline(df.delivery_days.median(), color="green", ls=":", label=f"Median {df.delivery_days.median():.2f}")
ax.set(title="Distribution of Delivery Time", xlabel="Delivery time (days)", ylabel="Shipments"); ax.legend()
plt.tight_layout(); plt.savefig("charts/c1_hist.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,4))
sns.boxplot(data=df, x="carrier", y="delivery_days", hue="weather", order=["Carrier X","Carrier Y","Carrier Z"], hue_order=["Clear","Rain","Storm"], ax=ax); ax.legend(title="Weather", loc="upper left", ncol=3, fontsize=8)
ax.set(title="Delivery Time by Carrier and Weather", xlabel="", ylabel="Delivery time (days)")
plt.tight_layout(); plt.savefig("charts/c2_box.png", dpi=150); plt.close()

fig, ax1 = plt.subplots(figsize=(7,4))
d = df.groupby("ship_date").agg(n=("shipment_id","count"), otd=("is_late", lambda s: 100*(1-s.mean())))
ax1.bar(d.index, d.n, color="#BDD7EE", label="Daily shipments"); ax1.set_ylabel("Shipments")
ax2 = ax1.twinx(); ax2.plot(d.index, d.otd.rolling(7).mean(), color="#C00000", lw=2, label="On-time % (7-day avg)")
ax2.set_ylabel("On-time delivery (%)"); ax2.grid(False)
ax1.set_title("Shipment Volume and On-Time Rate Over Time"); fig.autofmt_xdate()
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax1.legend(h1+h2,l1+l2,loc="upper center",bbox_to_anchor=(0.5,-0.25),ncol=2,frameon=False)
plt.tight_layout(); plt.savefig("charts/c3_trend.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,4.2))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="transport_cost", hue="vehicle", alpha=.6, s=18, ax=ax)
xs = np.linspace(df.distance_km.min(), df.distance_km.max(), 50); ax.plot(xs, np.polyval(sl, xs), color="black", lw=1.5, label="Linear fit")
ax.set(title="Transport Cost vs Distance", xlabel="Distance (km)", ylabel="Transport cost"); ax.legend()
plt.tight_layout(); plt.savefig("charts/c4_scatter.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6,4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax); ax.set_title("Correlation Matrix")
plt.tight_layout(); plt.savefig("charts/c5_corr.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.5,4))
sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label":"Late shipments (%)"})
ax.set_title("Late-Delivery Rate (%) by Carrier and Weather"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("charts/c6_heat.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,4))
order = df.groupby("region").cost_per_kg.median().sort_values().index
sns.barplot(data=df, x="region", y="cost_per_kg", order=order, estimator=np.median, errorbar=None, ax=ax, color="#2E75B6")
ax.set(title="Median Cost per kg by Region", xlabel="", ylabel="Cost per kg")
plt.tight_layout(); plt.savefig("charts/c7_bar.png", dpi=150); plt.close()
