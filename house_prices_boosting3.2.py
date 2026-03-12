#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
House Prices (home-data-for-ml-course) - Boosting Baseline (CPU-first) v3.0
==========================================================================
Key updates vs 2.0:
- CPU-first (no GPU path)
- StratifiedKFold (bucketized y_log) for more stable CV
- Better CatBoost ergonomics: progress output, no file writes, capped threads
- Default behavior: if use_catboost=true -> use CatBoost-only prediction (recommended)
- Optional blend: enable with blend=true; supports auto weight search on OOF

Usage (Server Command Examples)
===============================

多种子求平均使用实例：
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 42 --use_catboost true --plot true
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 2024 --use_catboost true --plot true
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 3407 --use_catboost true --plot true
cd ./output
python submission_average.py


(Recommended) CatBoost 单模（更强、更稳）
  python house_prices_boosting3.0.py --data_dir ./raw_data --out_dir ./output --use_catboost true

快速迭代（3折，默认）+ 画图
  python house_prices_boosting3.0.py --data_dir ./raw_data --out_dir ./output --use_catboost true --plot true

最终确认（5折）
  python house_prices_boosting3.0.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 42 --use_catboost true

开启融合（不推荐手拍权重；默认自动找最优权重）
  python house_prices_boosting3.0.py --data_dir ./raw_data --out_dir ./output --use_catboost true --blend true

若你仍想固定权重（w_cb = 0.40）
  python house_prices_boosting3.0.py --data_dir ./raw_data --out_dir ./output --use_catboost true --blend true --auto_blend false --lgb_weight 0.60 --cb_weight 0.40

输出结果：
- ./output/submission_seed_{seed}.csv (default)
- ./output/progress_seed_{seed}.png（若 plot=true，default）
"""

import os
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# -----------------------------
# CLI helpers: support key=value
# -----------------------------
def parse_kv_args(argv):
    """
    Support calling:
      python xxx.py data_dir=./raw_data out_dir=./output use_catboost=true plot=true
    """
    new_argv = [argv[0]]
    for arg in argv[1:]:
        if "=" in arg and not arg.startswith("--"):
            k, v = arg.split("=", 1)
            new_argv.extend(["--" + k, v])
        else:
            new_argv.append(arg)
    return new_argv


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Config:
    data_dir: str
    out_dir: str
    folds: int
    seed: int

    use_catboost: bool
    blend: bool
    auto_blend: bool
    blend_step: float
    lgb_weight: float
    cb_weight: float

    plot: bool
    plot_every: int

    cb_threads: int
    cb_verbose: int

    no_seed_suffix: bool


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Simple "Animator-like" plotter
# -----------------------------
class ProgressPlotter:
    def __init__(self, out_dir: str, enabled: bool = True, suffix: str = ""):
        self.enabled = enabled
        self.suffix = suffix
        self.out_path = os.path.join(out_dir, f"progress{suffix}.png")
        self.history = {}

        if not self.enabled:
            self.plt = None
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
        except Exception:
            self.enabled = False
            self.plt = None

    def update(self, key: str, iters: list, rmses: list):
        if not self.enabled:
            return
        self.history[key] = (iters, rmses)
        self._save()

    def _save(self):
        plt = self.plt
        if plt is None:
            return
        plt.figure()
        for key, (xs, ys) in self.history.items():
            if len(xs) == 0:
                continue
            plt.plot(xs, ys, label=key)
        plt.xlabel("iteration")
        plt.ylabel("RMSE (log1p)")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.out_path, dpi=160)
        plt.close()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea"]:
        if col not in df.columns:
            df[col] = 0

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    for col in ["YrSold", "YearBuilt", "YearRemodAdd"]:
        if col not in df.columns:
            df[col] = np.nan
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    for col in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if col not in df.columns:
            df[col] = 0
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    for col in porch_cols:
        if col not in df.columns:
            df[col] = 0
    df["TotalPorchSF"] = sum(df[c].fillna(0) for c in porch_cols)

    df["HasPool"] = (df.get("PoolArea", 0).fillna(0) > 0).astype(int)
    df["HasGarage"] = (df.get("GarageArea", 0).fillna(0) > 0).astype(int)
    df["HasBasement"] = (df.get("TotalBsmtSF", 0).fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0).fillna(0) > 0).astype(int)

    if "OverallQual" in df.columns:
        df["Qual_x_LivArea"] = df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)

    # Ratio / density features
    if "TotRmsAbvGrd" in df.columns:
        df["LivArea_per_Room"] = _safe_div(df["GrLivArea"].fillna(0), df["TotRmsAbvGrd"].fillna(0))
    if "BedroomAbvGr" in df.columns:
        df["Bath_per_Bed"] = _safe_div(df["TotalBathrooms"].fillna(0), df["BedroomAbvGr"].fillna(0))
    df["Bsmt_ratio"] = _safe_div(df["TotalBsmtSF"].fillna(0), df["TotalSF"].fillna(0))

    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()

    # Classic outliers: huge living area but low price
    if "GrLivArea" in train.columns and "SalePrice" in train.columns:
        train = train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index)

    y = np.log1p(train["SalePrice"].values)
    train_x = train.drop(columns=["SalePrice"])

    all_df = pd.concat([train_x, test], axis=0, ignore_index=True)
    all_df = add_features(all_df)

    none_cols = [
        "Alley", "PoolQC", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    for c in none_cols:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype("object").fillna("None")

    if "MSSubClass" in all_df.columns:
        all_df["MSSubClass"] = all_df["MSSubClass"].astype("object")

    # Categorical
    cat_cols = [c for c in all_df.columns if all_df[c].dtype == "object"]
    for c in cat_cols:
        all_df[c] = all_df[c].fillna("Missing").astype("category")

    # Numeric imputation
    num_cols = [c for c in all_df.columns if c not in cat_cols]
    for c in num_cols:
        if all_df[c].isna().any():
            all_df[c] = all_df[c].fillna(all_df[c].median())

    # Feature-level skew log1p (non-negative only)
    skew_vals = all_df[num_cols].skew(numeric_only=True).sort_values(ascending=False)
    skew_cols = [c for c in skew_vals.index if abs(skew_vals[c]) > 0.75]
    for c in skew_cols:
        if (all_df[c] >= 0).all():
            all_df[c] = np.log1p(all_df[c])

    X_train = all_df.iloc[: len(train), :].copy()
    X_test = all_df.iloc[len(train) :, :].copy()

    train_ids = train["Id"].values
    test_ids = test["Id"].values

    return X_train, y, X_test, train_ids, test_ids, cat_cols


def make_stratified_folds(y_log, n_splits, seed):
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(np.zeros(len(y_log)), bins)


def train_lgbm_cv(X_train, y, X_test, cat_cols, folds, seed, plotter: ProgressPlotter, plot_every: int):
    import lightgbm as lgb
    from tqdm import tqdm

    oof = np.zeros(len(X_train), dtype=float)
    pred_test = np.zeros(len(X_test), dtype=float)

    # CPU fast-iteration & stable params
    params = dict(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        objective="regression",
        verbosity=-1,
    )

    print("\n==== LightGBM CV ====")
    print("[LGBM] device_type = cpu")

    split_iter = make_stratified_folds(y, n_splits=folds, seed=seed)

    for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
        X_tr, X_va = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_va[c] = X_va[c].astype("category")

        model = lgb.LGBMRegressor(**params)
        evals_result = {}

        pbar = tqdm(
            total=params["n_estimators"],
            desc=f"LGBM fold {fold}/{folds}",
            ncols=100,
            mininterval=1.0,
            miniters=50,
        )
        last_it = 0

        def _cb_progress(env):
            nonlocal last_it
            it = env.iteration + 1
            if it > params["n_estimators"]:
                return
            delta = it - last_it
            if delta >= 50 or it == params["n_estimators"]:
                pbar.update(delta)
                last_it = it
                try:
                    rm = env.evaluation_result_list[0][2]
                    pbar.set_postfix({"rmse": f"{rm:.6f}"})
                except Exception:
                    pass

                if plotter and plotter.enabled and (it % plot_every == 0):
                    try:
                        ys = evals_result["valid_0"]["rmse"]
                        xs = list(range(1, len(ys) + 1))
                        plotter.update(f"LGBM_fold{fold}", xs, ys)
                    except Exception:
                        pass

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            categorical_feature=cat_cols,
            callbacks=[
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                _cb_progress,
            ],
        )
        pbar.close()

        best_iter = model.best_iteration_
        oof[va_idx] = model.predict(X_va, num_iteration=best_iter)
        pred_test += model.predict(X_test, num_iteration=best_iter) / folds

        fold_rm = rmse(y_va, oof[va_idx])
        print(f"[LGBM] Fold {fold}/{folds} RMSE(log1p): {fold_rm:.6f}  best_iter={best_iter}")

        if plotter and plotter.enabled:
            try:
                ys = evals_result["valid_0"]["rmse"]
                xs = list(range(1, len(ys) + 1))
                plotter.update(f"LGBM_fold{fold}", xs, ys)
            except Exception:
                pass

    full_rm = rmse(y, oof)
    print(f"[LGBM] CV RMSE(log1p): {full_rm:.6f}")
    return oof, pred_test, full_rm


def train_catboost_cv(X_train, y, X_test, cat_cols, folds, seed, plotter: ProgressPlotter, cb_threads: int, cb_verbose: int):
    from catboost import CatBoostRegressor, Pool

    oof = np.zeros(len(X_train), dtype=float)
    pred_test = np.zeros(len(X_test), dtype=float)

    # CPU-friendly + visible progress + avoid disk writes
    params = dict(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        iterations=2000,
        random_seed=seed,
        eval_metric="RMSE",
        task_type="CPU",
        thread_count=cb_threads,
        allow_writing_files=False,   # important on some servers
        verbose=cb_verbose,          # print every N iterations
    )

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    print("\n==== CatBoost CV ====")
    print(f"[CB] task_type = CPU | thread_count={cb_threads} | verbose={cb_verbose}")

    split_iter = make_stratified_folds(y, n_splits=folds, seed=seed)

    for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
        print(f"[CB] fold {fold}/{folds} ...")
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        va_pool = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True, early_stopping_rounds=200)

        oof[va_idx] = model.predict(X_va)
        pred_test += model.predict(X_test) / folds

        fold_rm = rmse(y_va, oof[va_idx])
        print(f"[CB]   Fold {fold}/{folds} RMSE(log1p): {fold_rm:.6f}  best_iter={model.get_best_iteration()}")

        if plotter and plotter.enabled:
            try:
                ev = model.get_evals_result()
                ys = ev["validation"]["RMSE"]
                xs = list(range(1, len(ys) + 1))
                plotter.update(f"CB_fold{fold}", xs, ys)
            except Exception:
                pass

    full_rm = rmse(y, oof)
    print(f"[CB]   CV RMSE(log1p): {full_rm:.6f}")
    return oof, pred_test, full_rm


def search_best_blend_weight(y_log, lgb_oof, cb_oof, step=0.01):
    """
    Find best w (for CatBoost) minimizing RMSE on OOF:
      blend = (1-w)*lgb + w*cb
    """
    best_w, best_rm = 0.0, float("inf")
    ws = np.arange(0.0, 1.0 + 1e-12, step)
    for w in ws:
        oof = (1.0 - w) * lgb_oof + w * cb_oof
        r = rmse(y_log, oof)
        if r < best_rm:
            best_rm, best_w = r, float(w)
    return best_w, best_rm


def main():
    import sys
    sys.argv = parse_kv_args(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--folds", type=int, default=3, help="fast: 3 | final: 5")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_catboost", type=str2bool, default=False, help="train CatBoost")
    parser.add_argument("--blend", type=str2bool, default=False, help="blend LGBM + CatBoost (requires use_catboost=true)")
    parser.add_argument("--auto_blend", type=str2bool, default=True, help="auto search best blend weight on OOF")
    parser.add_argument("--blend_step", type=float, default=0.01, help="step for auto blend weight search")

    parser.add_argument("--lgb_weight", type=float, default=0.6, help="manual blend weight for LGBM (when auto_blend=false)")
    parser.add_argument("--cb_weight", type=float, default=0.4, help="manual blend weight for CatBoost (when auto_blend=false)")

    parser.add_argument("--plot", type=str2bool, default=False, help="save progress curve to out_dir/progress_seed_{seed}.png")
    parser.add_argument("--plot_every", type=int, default=200)

    parser.add_argument("--cb_threads", type=int, default=0, help="0 -> auto cap to 8 for shared servers")
    parser.add_argument("--cb_verbose", type=int, default=200, help="CatBoost prints every N iterations")
    parser.add_argument("--no_seed_suffix", type=str2bool, default=False, help="do not add _seed_{seed} suffix to output filenames")

    args = parser.parse_args()

    cb_threads = args.cb_threads
    if cb_threads <= 0:
        cb_threads = min(8, os.cpu_count() or 8)

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        folds=args.folds,
        seed=args.seed,

        use_catboost=args.use_catboost,
        blend=args.blend,
        auto_blend=args.auto_blend,
        blend_step=args.blend_step,
        lgb_weight=args.lgb_weight,
        cb_weight=args.cb_weight,

        plot=args.plot,
        plot_every=args.plot_every,

        cb_threads=cb_threads,
        cb_verbose=args.cb_verbose,
        no_seed_suffix=args.no_seed_suffix,
    )

    seed_suffix = "" if cfg.no_seed_suffix else f"_seed_{cfg.seed}"

    out_dir_run = cfg.out_dir
    os.makedirs(out_dir_run, exist_ok=True)
    plotter = ProgressPlotter(out_dir_run, enabled=cfg.plot, suffix=seed_suffix)

    train_path = os.path.join(cfg.data_dir, "train.csv")
    test_path = os.path.join(cfg.data_dir, "test.csv")
    sub_path = os.path.join(cfg.data_dir, "sample_submission.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(sub_path)):
        raise FileNotFoundError("Missing files: train.csv/test.csv/sample_submission.csv in --data_dir")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path)

    X_train, y_log, X_test, train_ids, test_ids, cat_cols = preprocess(train, test)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"#Categorical columns: {len(cat_cols)}")
    if plotter.enabled:
        print(f"[Plot] Saving curve to: {os.path.join(out_dir_run, f'progress{seed_suffix}.png')}")

    # Decide what to train
    want_cb = cfg.use_catboost
    want_blend = cfg.blend and cfg.use_catboost
    want_lgb = (not want_cb) or want_blend  # LGB needed only if no CB or if blending

    lgb_oof = lgb_test_pred = None
    cb_oof = cb_test_pred = None

    if want_lgb:
        lgb_oof, lgb_test_pred, _ = train_lgbm_cv(
            X_train, y_log, X_test, cat_cols,
            folds=cfg.folds, seed=cfg.seed,
            plotter=plotter, plot_every=cfg.plot_every
        )

    if want_cb:
        cb_oof, cb_test_pred, _ = train_catboost_cv(
            X_train, y_log, X_test, cat_cols,
            folds=cfg.folds, seed=cfg.seed,
            plotter=plotter,
            cb_threads=cfg.cb_threads,
            cb_verbose=cfg.cb_verbose,
        )

    # Build final prediction
    used_models = []
    if want_blend:
        # Auto weight search on OOF
        if cfg.auto_blend:
            w_cb, best_rm = search_best_blend_weight(y_log, lgb_oof, cb_oof, step=cfg.blend_step)
            w_lgb = 1.0 - w_cb
            final_test_pred = w_lgb * lgb_test_pred + w_cb * cb_test_pred
            used_models = [f"LightGBM(w={w_lgb:.2f})", f"CatBoost(w={w_cb:.2f})"]
            print(f"[BLEND-AUTO] Best w_cb={w_cb:.2f}, w_lgb={w_lgb:.2f} | CV RMSE(log1p): {best_rm:.6f}")
        else:
            w1, w2 = cfg.lgb_weight, cfg.cb_weight
            s = w1 + w2
            w1, w2 = w1 / s, w2 / s
            final_test_pred = w1 * lgb_test_pred + w2 * cb_test_pred
            blend_oof = w1 * lgb_oof + w2 * cb_oof
            blend_cv = rmse(y_log, blend_oof)
            used_models = [f"LightGBM(w={w1:.2f})", f"CatBoost(w={w2:.2f})"]
            print(f"[BLEND] CV RMSE(log1p): {blend_cv:.6f}  (LGBM {w1:.2f} + CB {w2:.2f})")

    elif want_cb:
        # Default recommended path: CatBoost-only
        final_test_pred = cb_test_pred
        used_models = ["CatBoost"]
    else:
        # LightGBM-only
        final_test_pred = lgb_test_pred
        used_models = ["LightGBM"]

    # Submission
    submission = sample_sub.copy()
    submission["Id"] = test_ids
    submission["SalePrice"] = np.expm1(final_test_pred)

    out_path = os.path.join(out_dir_run, f"submission{seed_suffix}.csv")
    submission.to_csv(out_path, index=False)

    print("-" * 80)
    print("Finished.")
    print("Models:", " + ".join(used_models))
    print("Saved submission:", out_path)


if __name__ == "__main__":
    main()
