import numpy as np
import pandas as pd
from genki_backtest import BASE_ALLOCATION, VARIANT_BUDGETS, historical_proxy, make_tilt_target, normalized_base, run_audit, simulate


def sample_prices():
    dates=pd.bdate_range("2018-01-01",periods=1600); rng=np.random.default_rng(88); n=len(BASE_ALLOCATION)
    returns=rng.normal(.00015,.009,(len(dates),n)); returns[:,0]+=np.sin(np.arange(len(dates))/130)*.0008; returns[:,2]-=np.sin(np.arange(len(dates))/130)*.0006
    return pd.DataFrame(100*np.cumprod(1+returns,axis=0),index=dates,columns=BASE_ALLOCATION)


def test_no_tilt_variant_matches_base_and_periods_match():
    p=sample_prices(); scores,levels=historical_proxy(p); levels["action_level"]=4
    base=simulate(p,"Base Strategy",scores,levels,.2,5,0)
    assert not base.events and base.equity.index.equals(base.gross_equity.index)


def test_tilt_conserves_total_respects_budget_and_action_one():
    p=sample_prices(); base=normalized_base(); scores=pd.Series({g:i for i,g in enumerate([*__import__('genki_backtest').ROLE_GROUPS])})
    unchanged,info=make_tilt_target(base,p,scores,1,.10); assert unchanged.equals(base) and info["budget"]==0
    tilted,info=make_tilt_target(base,p,scores,4,.10); assert np.isclose(tilted.sum(),base.sum()); assert info["budget"]<=.10
    assert np.isclose((tilted-base).clip(lower=0).sum(), -(tilted-base).clip(upper=0).sum())


def test_conservative_events_not_more_than_standard_and_costs_reduce_equity():
    p=sample_prices(); scores,levels=historical_proxy(p); levels["action_level"]=2
    con=simulate(p,"GENKI Conservative",scores,levels,.2,5,0); std=simulate(p,"GENKI Standard",scores,levels,.2,5,0)
    assert len(con.events)<=len(std.events); assert std.equity.iloc[-1] <= std.gross_equity.iloc[-1]


def test_required_artifacts_and_japanese_report(tmp_path):
    tables=run_audit(sample_prices(),tmp_path)
    assert tables["equity_curves"].dropna().apply(lambda x:x.first_valid_index()).nunique()==1
    for f in ["genki_summary_report.md","genki_metrics.csv","genki_annual_returns.csv","tilt_events.csv","equity_curves.csv","equity_curve.png","drawdown_curve.png"]: assert (tmp_path/f).is_file()
    assert "【結論】" in (tmp_path/"genki_summary_report.md").read_text(encoding="utf-8")
