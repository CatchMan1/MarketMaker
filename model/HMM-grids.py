# 导包
import json
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class HMMMarketStrategy:
    """Encapsulates the preprocessing, modelling and evaluation pipeline."""

    # 最多设置7种状态
    COLORS = ['#ff7f0e', '#1f77b4', "#8CBF87", "#8D2F25", "#909291", "#4E1945", "#B7B7Eb"]

    def __init__(self,params):
        self.data_path = params['data_path']
        self.window = params['window']
        self.model_features = params['model_features']
        self.train_end = pd.to_datetime(params['train_end'])
        self.test_start = pd.to_datetime(params['test_start'])
        self.return_column = params['return_column']

        self.n_components = params['n_components']
        self.covariance_type = params['covariance_type']
        self.n_iter = params['n_iter']
        self.random_state = params['random_state']
        self.log_dir = Path(params['log_dir'])
        self.save_plots = params['save_plots']

        self.df_min = None
        self.df_train_norm = None
        self.df_test_norm = None
        self.df_train_plot = None
        self.df_test_plot = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        columns_to_drop = [col for col in ['minute', 'pct', 'slippage'] if col in df.columns]
        self.df_min = df.drop(columns=columns_to_drop)

    def rolling_standardize(self):
        rolling_mean = self.df_min[self.model_features].rolling(window=self.window).mean()
        rolling_std = self.df_min[self.model_features].rolling(window=self.window).std()

        df_feature_rolling = (self.df_min[self.model_features] - rolling_mean) / rolling_std
        df_feature_rolling = df_feature_rolling.dropna()

        self.df_train_norm = df_feature_rolling.loc[:self.train_end]
        self.df_test_norm = df_feature_rolling.loc[self.test_start:]
        self.df_train_plot = self.df_min.loc[self.df_train_norm.index].copy()
        self.df_test_plot = self.df_min.loc[self.df_test_norm.index].copy()

    def fit_model(self):
        X_train = self.df_train_norm.values
        X_test = self.df_test_norm.values

        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_train)

        self.df_train_plot['state'] = self.model.predict(X_train)
        self.df_test_plot['state'] = self.model.predict(X_test)

    def _sanitize_features_label(self):
        joined = "_".join(self.model_features)
        safe = joined.replace('/', '_').replace(' ', '')
        return safe

    def _plot_state_contribution(self, df_plot, title, save_path=None):
        plt.figure(figsize=(15, 8))
        for i in range(self.n_components):
            df_sample = df_plot.copy()
            df_sample['signal'] = (df_plot['state'] == i).shift(1).fillna(0).astype(float)

            f_vals = df_sample[self.return_column].astype(float).values
            signal = df_sample['signal'].astype(float).values
            strat_ret = f_vals * signal
            strat_ret[0] = f_vals[0]

            strategy_cum = np.exp(np.insert(np.cumsum(strat_ret), 0, 0))
            time_delta = df_sample.index[1] - df_sample.index[0]
            plot_dates = [df_sample.index[0] - time_delta] + list(df_sample.index)

            color = self.COLORS[i % len(self.COLORS)]
            plt.plot(plot_dates, strategy_cum, color=color, label=f'State {i} Performance')

        plt.axhline(1, color='red', linestyle='--', alpha=0.5)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def calculate_mdd(cum_series):
        nav = np.array(cum_series)
        peaks = np.maximum.accumulate(nav)
        drawdowns = (nav - peaks) / peaks
        idx = np.argmin(drawdowns)
        return np.min(drawdowns), idx

    def prepare(self):
        self.load_data()
        self.rolling_standardize()
        self.fit_model()

    def calculate_state_final_nav(self, df_plot):
        final_nav = {}
        for i in range(self.n_components):
            df_sample = df_plot.copy()
            signal = (df_sample['state'] == i).shift(1).fillna(0).astype(float)
            f_values = df_sample[self.return_column].astype(float).values
            strat_ret = f_values * signal
            strat_ret[0] = f_values[0]
            strategy_cum = np.exp(np.insert(np.cumsum(strat_ret), 0, 0))
            final_nav[i] = strategy_cum[-1]
        return final_nav

    def determine_positive_states(self):
        state_nav = self.calculate_state_final_nav(self.df_train_plot)
        return [state for state, nav in state_nav.items() if nav > 1]

    def run_backtest(self, df_plot, title, use_states, save_path=None, plot=True, verbose=True):
        df = df_plot.copy()
        df.index = pd.to_datetime(df.index)
        signal = df['state'].isin(use_states).shift(1).fillna(0).astype(float)

        f_values = df[self.return_column].astype(float).values

        strat_ret = f_values * signal.values
        strat_ret[0] = f_values[0]

        market_cum = np.exp(np.insert(np.cumsum(f_values), 0, 0))
        strategy_cum = np.exp(np.insert(np.cumsum(strat_ret), 0, 0))
        time_delta = df.index[1] - df.index[0]
        plot_dates = [df.index[0] - time_delta] + list(df.index)

        total_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
        final_nav = strategy_cum[-1]
        annual_ret = ((final_nav ** (252 / total_days)) - 1) * 100

        mdd_market_val, mdd_market_idx = self.calculate_mdd(market_cum)
        mdd_strategy_val, mdd_strategy_idx = self.calculate_mdd(strategy_cum)
        mdd_market_date = plot_dates[mdd_market_idx]
        mdd_strategy_date = plot_dates[mdd_strategy_idx]

        if verbose:
            print(f"【市场基准】最大回撤: {mdd_market_val:.2%} | 发生时间: {mdd_market_date}")
            print(f"【HMM策略】最大回撤: {mdd_strategy_val:.2%} | 发生时间: {mdd_strategy_date}")

        metrics = {
            'final_nav': final_nav,
            'annual_return': annual_ret,
            'market_mdd': mdd_market_val,
            'strategy_mdd': mdd_strategy_val,
            'invest_states': use_states,
        }

        if plot:
            plt.figure(figsize=(14, 7), dpi=100)
            plt.plot(plot_dates, market_cum, color='#1f77b4', label='Market (Benchmark)', alpha=0.8)
            plt.plot(plot_dates, strategy_cum, color='#ff7f0e', label='HMM Strategy', linewidth=2.5)

            market_peak = np.maximum.accumulate(market_cum)
            strategy_peak = np.maximum.accumulate(strategy_cum)
            plt.fill_between(plot_dates, market_cum, market_peak, color='blue', alpha=0.1, label='Market-Drawdown')
            plt.fill_between(plot_dates, strategy_cum, strategy_peak, color='red', alpha=0.1, label='HMM-Drawdown')

            plt.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            plt.title(f"{title}\nFinal Return: {annual_ret:.2f}% | Invest States: {use_states}", fontsize=14)

            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper left')

            plt.tight_layout()
            plt.scatter(
                mdd_market_date,
                market_cum[mdd_market_idx],
                color='blue',
                s=100,
                marker='x',
                label=f'Market MaxDD: {mdd_market_val:.2%}',
                zorder=5,
            )
            plt.scatter(
                mdd_strategy_date,
                strategy_cum[mdd_strategy_idx],
                color='red',
                s=100,
                marker='o',
                label=f'HMM MaxDD: {mdd_strategy_val:.2%}',
                zorder=5,
            )
            plt.annotate(
                f'Market MaxDD\n{mdd_market_val:.2%}',
                xy=(mdd_market_date, market_cum[mdd_market_idx]),
                xytext=(20, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='blue'),
            )
            plt.annotate(
                f'HMM MaxDD\n{mdd_strategy_val:.2%}',
                xy=(mdd_strategy_date, strategy_cum[mdd_strategy_idx]),
                xytext=(20, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'),
            )
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            plt.close('all')

        return metrics

    def run(self, backtrace_params):
        self.prepare()

        feature_label = self._sanitize_features_label()
        combo_dir = self.log_dir / feature_label
        if self.save_plots:
            combo_dir.mkdir(parents=True, exist_ok=True)
        in_plot = combo_dir / f"In_sample_{self.n_components}.png"
        out_plot = combo_dir / f"output_sample_{self.n_components}.png"

        self._plot_state_contribution(
            self.df_train_plot,
            "HMM Individual State Contribution (In-of-Sample)",
            save_path=in_plot if self.save_plots else None,
        )
        self._plot_state_contribution(
            self.df_test_plot,
            "HMM Individual State Contribution (Out-of-Sample)",
            save_path=out_plot if self.save_plots else None,
        )

        state_set = backtrace_params['invest_states']
        run_backtests = backtrace_params['run_backtests']
        if run_backtests:
            eva_in = combo_dir / f"Eva_In_sample_{self.n_components}.png"
            eva_out = combo_dir / f"Eva_out_sample_{self.n_components}.png"
            self.run_backtest(
                self.df_train_plot,
                "In-Sample Evaluation",
                state_set,
                save_path=eva_in,
                plot=True,
            )
            self.run_backtest(
                self.df_test_plot,
                "Out-of-Sample Evaluation",
                state_set,
                save_path=eva_out,
                plot=True,
            )


def generate_feature_combinations(candidate_features, min_len=3, max_len=5):
    max_len = min(max_len, len(candidate_features))
    combinations = []
    for size in range(min_len, max_len + 1):
        combinations.extend([list(combo) for combo in itertools.combinations(candidate_features, size)])
    return combinations


def run_grid_search(base_params, feature_candidates, component_choices, min_feat=3, max_feat=5):
    combo_list = generate_feature_combinations(feature_candidates, min_feat, max_feat)
    if not combo_list:
        raise ValueError("No feature combinations generated for grid search.")

    log_path = Path(base_params['log_dir'])
    log_path.mkdir(parents=True, exist_ok=True)

    best_result = {'final_nav': -np.inf}
    for combo in combo_list:
        for n_components in component_choices:
            params = base_params.copy()
            params['model_features'] = combo
            params['n_components'] = n_components
            params['save_plots'] = False

            strategy = HMMMarketStrategy(params)
            strategy.prepare()
            positive_states = strategy.determine_positive_states()
            if not positive_states:
                continue

            metrics = strategy.run_backtest(
                strategy.df_test_plot,
                "Grid Search Out-of-Sample",
                positive_states,
                plot=False,
                verbose=False,
            )
            combo_label = strategy._sanitize_features_label()
            final_nav = metrics['final_nav']
            print(
                f"Grid candidate {combo_label}, states={n_components}, invest_states={positive_states}, test final_nav={final_nav:.4f}"
            )

            if final_nav > best_result['final_nav']:
                best_result = {
                    'final_nav': final_nav,
                    'annual_return': metrics['annual_return'],
                    'feature_combo': list(combo),
                    'feature_label': combo_label,
                    'n_components': n_components,
                    'invest_states': positive_states,
                    'log_folder': str(log_path / combo_label),
                }

    if best_result['final_nav'] == -np.inf:
        print("Grid search did not find any positive-state combination.")
        return None

    best_file = log_path / "best_grid_result.json"
    best_file.write_text(json.dumps(best_result, ensure_ascii=False, indent=2))
    print(f"Best grid result stored at {best_file}")
    return best_result


FEATURE_CANDIDATES = [
    'vol_log_ret_1p',
    'vol_log_ret_5p',
    'close_log_ret_1p',
    'close_log_ret_5p',
    'intraday_range_log',
    'successive_increase_count',
    'successive_decrease_count',
]

BASE_PARAMS = {
    'data_path': '../data/kline_im_more.csv',
    'window': 336,
    'model_features': FEATURE_CANDIDATES,
    'train_end': '2024-09-30 15:00:00',
    'test_start': '2024-10-01 09:30:00',
    'return_column': 'close_log_ret_1p',
    'n_components': 4,
    'covariance_type': 'full',
    'n_iter': 1000,
    'random_state': 42,
    'log_dir': 'logs',
    'save_plots': False,
}

GRID_N_COMPONENTS = [3, 4, 5, 6]


if __name__ == "__main__":
    best_config = run_grid_search(BASE_PARAMS, FEATURE_CANDIDATES, GRID_N_COMPONENTS, min_feat=3, max_feat=4)
    if best_config:
        final_params = BASE_PARAMS.copy()
        final_params.update({
            'model_features': best_config['feature_combo'],
            'n_components': best_config['n_components'],
            'save_plots': True,
        })

        best_strategy = HMMMarketStrategy(final_params)
        best_strategy.run({
            'invest_states': best_config['invest_states'],
            'run_backtests': True,
        })