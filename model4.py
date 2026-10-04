import numpy as np, pandas as pd, matplotlib, json, warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.optimize import linprog
sns.set_theme(style="whitegrid", palette="deep")
R = {}

df = pd.read_csv("logistics_sim.csv", parse_dates=["ship_date"]).sort_values("ship_date").reset_index(drop=True)
df["dow"] = df["ship_date"].dt.dayofweek.astype(str)
df["is_weekend_hand"] = df["dow"].isin(["4","5"]).astype(int)
num = ["distance_km","weight_kg","is_weekend_hand"]
cat = ["carrier","region","priority","weather","warehouse"]
X, y = df[num+cat], df["delivery_days"]
cut = pd.Timestamp("2026-09-08")
tr, te = df.ship_date < cut, df.ship_date >= cut
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
R["n_train"], R["n_test"] = int(tr.sum()), int(te.sum())

pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
rmse = lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
models = {
 "Baseline (mean)": DummyRegressor(),
 "Linear Regression": LinearRegression(),
 "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
 "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
 "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
tscv = TimeSeriesSplit(n_splits=5)
rows = []
fitted = {}
for name, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)])
    cvr = -cross_val_score(p, Xtr, ytr, cv=tscv, scoring="neg_root_mean_squared_error")
    p.fit(Xtr, ytr); pr = p.predict(Xte); fitted[name] = p
    rows.append([name, cvr.mean(), cvr.std(), rmse(yte,pr), mean_absolute_error(yte,pr), r2_score(yte,pr)])
res = pd.DataFrame(rows, columns=["Model","CV RMSE","CV std","Test RMSE","Test MAE","Test R2"]).round(3)
print(res.to_string()); R["models"] = res.to_dict("records")

# Hyperparameter tuning: gradient boosting + random forest
gb_grid = {"m__n_estimators":[100,200,300], "m__max_depth":[2,3,4], "m__learning_rate":[0.03,0.05,0.1]}
gs = GridSearchCV(Pipeline([("pre",pre),("m",GradientBoostingRegressor(random_state=42))]), gb_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
pg = gs.predict(Xte)
R["gb_best"] = {k:(v if not isinstance(v,(np.integer,np.floating)) else float(v)) for k,v in gs.best_params_.items()}
R["gb_cv"] = float(-gs.best_score_); R["gb_tuned"] = dict(rmse=rmse(yte,pg), mae=mean_absolute_error(yte,pg), r2=r2_score(yte,pg))
rf_grid = {"m__n_estimators":[100,300], "m__max_depth":[4,8,None], "m__min_samples_leaf":[2,5,10]}
gr = GridSearchCV(Pipeline([("pre",pre),("m",RandomForestRegressor(random_state=42,n_jobs=-1))]), rf_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
prf = gr.predict(Xte)
R["rf_best"] = {k:(v if v is None or isinstance(v,(str,int)) else float(v)) for k,v in gr.best_params_.items()}
R["rf_cv"] = float(-gr.best_score_); R["rf_tuned"] = dict(rmse=rmse(yte,prf), mae=mean_absolute_error(yte,prf), r2=r2_score(yte,prf))
print(R["gb_best"], R["gb_cv"], R["gb_tuned"]); print(R["rf_best"], R["rf_cv"], R["rf_tuned"])

# Final model choice by CV RMSE among Linear / tuned GB / tuned RF
cands = {"Linear Regression":(res.set_index("Model").loc["Linear Regression","CV RMSE"], fitted["Linear Regression"]),
         "Tuned Gradient Boosting":(R["gb_cv"], gs.best_estimator_), "Tuned Random Forest":(R["rf_cv"], gr.best_estimator_)}
best_name = min(cands, key=lambda k: cands[k][0]); best = cands[best_name][1]; R["best_name"] = best_name
pb = best.predict(Xte)
R["best_test"] = dict(rmse=rmse(yte,pb), mae=mean_absolute_error(yte,pb), r2=r2_score(yte,pb))
print("BEST", best_name, R["best_test"])

# Permutation importance on test (grouped by original features, via pipeline)
pi = permutation_importance(best, Xte, yte, n_repeats=15, random_state=42, scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print(imp.round(3).to_dict()); R["imp"] = imp.round(3).to_dict()

# Charts
fig, ax = plt.subplots(figsize=(7,3.8))
rr = res.sort_values("Test RMSE", ascending=False)
ax.barh(rr["Model"], rr["Test RMSE"], color="#2E75B6")
for i,v in enumerate(rr["Test RMSE"]): ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set(title="Test RMSE by Model (lower is better)", xlabel="RMSE (days)"); plt.tight_layout(); plt.savefig("charts/m1_models.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5.5,5))
ax.scatter(yte, pb, s=10, alpha=.4, color="#2E75B6"); lim=[0.5, max(yte.max(), pb.max())+.2]
ax.plot(lim, lim, "r--", label="Perfect prediction"); ax.set(xlim=lim, ylim=lim, xlabel="Actual delivery time (days)", ylabel="Predicted (days)", title=f"Actual vs Predicted ({best_name})"); ax.legend()
plt.tight_layout(); plt.savefig("charts/m2_actual_pred.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,3.8))
imp.sort_values().plot.barh(ax=ax, color="#2E75B6"); ax.set(title="Permutation Feature Importance", xlabel="Increase in RMSE when feature is shuffled (days)")
plt.tight_layout(); plt.savefig("charts/m3_imp.png", dpi=150); plt.close()

resid = yte - pb
R["resid_mean"], R["resid_std"] = float(resid.mean()), float(resid.std())
fig, ax = plt.subplots(1,2,figsize=(8,3.6))
sns.histplot(resid, kde=True, ax=ax[0], color="#2E75B6"); ax[0].set(title="Residual distribution", xlabel="Actual - predicted (days)")
ax[1].scatter(pb, resid, s=8, alpha=.4); ax[1].axhline(0, color="r", ls="--"); ax[1].set(title="Residuals vs predicted", xlabel="Predicted (days)", ylabel="Residual")
plt.tight_layout(); plt.savefig("charts/m4_resid.png", dpi=150); plt.close()

# ---------- Optimisation 1: data-driven delivery promise ----------
oof = cross_val_predict(best, Xtr, ytr, cv=tscv) if False else None
# out-of-fold residuals on training period (expanding window) for the later folds
resid_tr = []
for a,b in tscv.split(Xtr):
    m = __import__("sklearn").base.clone(best).fit(Xtr.iloc[a], ytr.iloc[a]); resid_tr.append(ytr.iloc[b]-m.predict(Xtr.iloc[b]))
resid_tr = pd.concat(resid_tr)
q90 = float(resid_tr.quantile(.90)); q95 = float(resid_tr.quantile(.95)); R["q90"], R["q95"] = q90, q95
te_df = df[te].copy(); te_df["pred"] = pb
cur_otd = float((te_df.delivery_days <= te_df.planned_days).mean()*100); cur_days = float(te_df.planned_days.mean())
R["cur"] = dict(otd=cur_otd, promise=cur_days)
opts = {}
for lab,q in [("pred + q90 buffer", q90), ("pred + q95 buffer", q95)]:
    prom = te_df.pred + q
    opts[lab] = dict(otd=float((te_df.delivery_days <= prom).mean()*100), promise=float(prom.mean()))
R["promise"] = opts
# by weather for the q90 policy
prom = te_df.pred + q90
bw = pd.DataFrame({"weather":te_df.weather, "cur_prom":te_df.planned_days, "new_prom":prom, "cur_ok":te_df.delivery_days<=te_df.planned_days, "new_ok":te_df.delivery_days<=prom})
byw = bw.groupby("weather").agg(n=("cur_ok","size"), cur_prom=("cur_prom","mean"), new_prom=("new_prom","mean"), cur_otd=("cur_ok","mean"), new_otd=("new_ok","mean")).round(3)
byw[["cur_otd","new_otd"]] *= 100; print(byw.round(1)); R["byw"] = byw.round(2).reset_index().to_dict("records")
print(R["cur"], opts, q90, q95)

fig, ax = plt.subplots(figsize=(7,3.8))
x = np.arange(len(byw)); w=.35; order=["Clear","Rain","Storm"]; b2=byw.loc[order]
ax.bar(x-w/2, b2.cur_prom, w, label="Current promise", color="#BDD7EE"); ax.bar(x+w/2, b2.new_prom, w, label="Model-based promise (pred + q90)", color="#2E75B6")
for i,(a,b) in enumerate(zip(b2.cur_prom,b2.new_prom)): ax.text(i-w/2,a+.03,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+.03,f"{b:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(order); ax.set(title="Average promised delivery time by weather", ylabel="Days"); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig("charts/m5_promise.png", dpi=150); plt.close()

# ---------- Optimisation 2: carrier allocation LP ----------
LATE_PEN = 50.0
trd = df[tr]; carriers = ["Carrier X","Carrier Y","Carrier Z"]; weathers = ["Clear","Rain","Storm"]
unit = trd.groupby(["weather","carrier"]).agg(cost=("transport_cost","mean"), late=("is_late","mean"))
unit["total"] = unit.cost + LATE_PEN*unit.late
print(unit.round(2))
vol = te_df.weather.value_counts().reindex(weathers); cur_share = te_df.groupby("weather").carrier.value_counts(normalize=True).unstack().reindex(weathers)[carriers]
cap = 0.50*len(te_df)                       # no carrier may carry more than 50% of volume
cvec = np.array([unit.loc[(w,c),"total"] for w in weathers for c in carriers])
A_eq = np.zeros((3,9)); b_eq = vol.values.astype(float)
for i in range(3): A_eq[i,3*i:3*i+3] = 1
A_ub = np.zeros((3,9)); b_ub = np.full(3, cap)
for j in range(3):
    for i in range(3): A_ub[j,3*i+j] = 1
sol = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None), method="highs")
xopt = sol.x.reshape(3,3)
cur_alloc = (cur_share.values * vol.values[:,None])
cur_cost = float((cur_alloc.flatten()*cvec).sum()); opt_cost = float(sol.fun)
R["lp"] = dict(cur_cost=cur_cost, opt_cost=opt_cost, saving_pct=100*(cur_cost-opt_cost)/cur_cost, per_ship_cur=cur_cost/len(te_df), per_ship_opt=opt_cost/len(te_df), n=len(te_df))
share_opt = pd.DataFrame(xopt/vol.values[:,None], index=weathers, columns=carriers)
print(share_opt.round(3)); print(cur_share.round(3)); print(R["lp"])
# expected late & cost for current vs optimised
lat = np.array([unit.loc[(w,c),"late"] for w in weathers for c in carriers]); cst = np.array([unit.loc[(w,c),"cost"] for w in weathers for c in carriers])
R["lp"].update(cur_late=float((cur_alloc.flatten()*lat).sum()), opt_late=float((xopt.flatten()*lat).sum()), cur_dir=float((cur_alloc.flatten()*cst).sum()), opt_dir=float((xopt.flatten()*cst).sum()))
R["share_opt"] = share_opt.round(3).to_dict(); R["share_cur"] = cur_share.round(3).to_dict(); R["unit"] = unit.round(3).reset_index().to_dict("records")
fig, axs = plt.subplots(1,2,figsize=(8,3.8), sharey=True)
cols = ["#2E75B6","#9DC3E6","#C00000"]
cur_share.plot.bar(stacked=True, ax=axs[0], color=cols, legend=False); axs[0].set(title="Current allocation", ylabel="Share of shipments", xlabel="")
share_opt.plot.bar(stacked=True, ax=axs[1], color=cols); axs[1].set(title="Optimised allocation (LP)", xlabel=""); axs[1].legend(fontsize=8, loc="lower right")
for a in axs: a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("charts/m6_alloc.png", dpi=150); plt.close()
json.dump(R, open("results4.json","w"), indent=1, default=str)
