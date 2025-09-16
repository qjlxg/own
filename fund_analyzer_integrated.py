```python
import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import akshare as ak
import random

class FundDataFetcher:
    """负责数据获取、清洗和缓存管理"""
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.risk_free_rate = self._get_risk_free_rate()
        self._web_headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
            ]),
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'http://fund.eastmoney.com/'
        }

    def _log(self, message: str):
        print(f"[数据获取] {message}")

    def _load_cache(self, force_refresh: bool = False) -> dict:
        if force_refresh or not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            last_updated = datetime.strptime(cache.get('last_updated'), '%Y-%m-%d')
            if (datetime.now() - last_updated).days > 3:
                self._log("缓存数据已过期，将重新获取")
                return {}
            self._log("使用有效缓存数据")
            return cache
        except Exception as e:
            self._log(f"缓存加载失败: {e}")
            return {}

    def _save_cache(self):
        self.cache['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)

    def _get_risk_free_rate(self) -> float:
        try:
            bond_data = ak.bond_zh_us_rate()
            risk_free_rate = bond_data[bond_data['item_name'] == '中国10年期国债']['value'].iloc[-1] / 100
            self._log(f"获取无风险利率：{risk_free_rate:.4f}")
            return risk_free_rate
        except Exception as e:
            self._log(f"获取无风险利率失败，默认0.018298: {e}")
            return 0.018298

    def get_market_sentiment(self):
        if 'market' in self.cache:
            self._log("使用缓存市场情绪数据")
            return self.cache['market']
        self._log("获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            csi500 = ak.stock_zh_index_daily_em(symbol="sz399905")
            index_data['date'] = pd.to_datetime(index_data['date'])
            csi500['date'] = pd.to_datetime(csi500['date'])
            last_week_data = index_data.iloc[-7:]
            last_week_csi500 = csi500.iloc[-7:]
            price_change = last_week_data['close'].iloc[-1] / last_week_data['close'].iloc[0] - 1
            csi500_change = last_week_csi500['close'].iloc[-1] / last_week_csi500['close'].iloc[0] - 1
            sentiment_score = 0.6 * price_change + 0.4 * csi500_change
            sentiment, trend = ('optimistic', 'bullish') if sentiment_score > 0.05 else ('pessimistic', 'bearish') if sentiment_score < -0.05 else ('neutral', 'neutral')
            market_data = {'sentiment': sentiment, 'trend': trend}
            self.cache['market'] = market_data
            self._save_cache()
            return market_data
        except Exception as e:
            self._log(f"市场情绪获取失败: {e}")
            return {'sentiment': 'neutral', 'trend': 'neutral'}

    def get_fund_data(self, fund_code: str, fund_name: str) -> dict:
        cache_key = f"{fund_code}_{fund_name}"
        if self.cache_data and cache_key in self.cache.get('fund_metrics', {}):
            return self.cache['fund_metrics'][cache_key]
        self._log(f"获取基金 {fund_code} 的实时数据...")
        for attempt in range(5):
            try:
                fund_data_ak = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                fund_data_ak['净值日期'] = pd.to_datetime(fund_data_ak['净值日期'])
                fund_data_ak.set_index('净值日期', inplace=True)
                fund_data_ak = fund_data_ak.dropna()
                if len(fund_data_ak) < 252:
                    raise ValueError("数据不足")
                returns = fund_data_ak['单位净值'].pct_change().dropna()
                annual_returns = returns.mean() * 252
                annual_volatility = returns.std() * (252**0.5)
                sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
                rolling_max = fund_data_ak['单位净值'].cummax()
                daily_drawdown = (fund_data_ak['单位净值'] - rolling_max) / rolling_max
                max_drawdown = daily_drawdown.min() * -1
                data = {
                    'latest_nav': float(fund_data_ak['单位净值'].iloc[-1]),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown)
                }
                if self.cache_data:
                    self.cache.setdefault('fund_metrics', {})[cache_key] = data
                    self._save_cache()
                return data
            except Exception as e:
                self._log(f"获取 {fund_code} 数据失败 (尝试 {attempt+1}/5): {e}")
                time.sleep(5)
        try:
            url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
            response = requests.get(url, headers=self._web_headers, timeout=10)
            data = json.loads(response.text.replace('jsonpgz(', '').replace(');', ''))
            data = {'latest_nav': float(data['dwjz']), 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            self.cache.setdefault('fund_metrics', {})[cache_key] = data
            self._save_cache()
            return data
        except Exception as e:
            self._log(f"雪球API获取 {fund_code} 失败: {e}")
            return {}

    def get_fund_manager_data(self, fund_code: str) -> dict:
        if self.cache_data and fund_code in self.cache.get('manager', {}):
            return self.cache['manager'][fund_code]
        self._log(f"获取基金 {fund_code} 经理数据...")
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        try:
            response = requests.get(manager_url, headers=self._web_headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='manager-table') or soup.find('table')
            if not table or len(table.find_all('tr')) < 2:
                return {}
            row = table.find_all('tr')[1]
            cols = row.find_all('td')
            if len(cols) < 5:
                return {}
            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            return_str = cols[4].text.strip()
            tenure_days = float(re.search(r'\d+', tenure_str).group()) if '天' in tenure_str else (
                float(re.search(r'\d+', tenure_str).group()) * 365 if '年' in tenure_str else np.nan)
            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', return_str).group()) if '%' in return_str else np.nan
            data = {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
            if self.cache_data:
                self.cache.setdefault('manager', {})[fund_code] = data
                self._save_cache()
            return data
        except Exception as e:
            self._log(f"经理数据获取失败: {e}")
            return {}

    def get_fund_holdings_data(self, fund_code: str) -> dict:
        if self.cache_data and fund_code in self.cache.get('holdings', {}):
            return self.cache['holdings'][fund_code]
        self._log(f"获取基金 {fund_code} 持仓数据...")
        holdings_url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
        try:
            response = requests.get(holdings_url, headers=self._web_headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            holdings = []
            sectors = []
            holdings_header = soup.find('h4', string=lambda t: t and '股票投资明细' in t)
            if holdings_header:
                table = holdings_header.find_next('table')
                if table:
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            holdings.append({
                                '股票代码': cols[1].text.strip(),
                                '股票名称': cols[2].text.strip(),
                                '占净值比例': float(cols[4].text.strip().replace('%', '')) if cols[4].text.strip() != '-' else 0
                            })
            sector_header = soup.find('h4', string=lambda t: t and '股票行业配置' in t)
            if sector_header:
                table = sector_header.find_next('table')
                if table:
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            sectors.append({
                                '行业名称': cols[0].text.strip(),
                                '占净值比例': float(cols[1].text.strip().replace('%', '')) if cols[1].text.strip() != '-' else 0
                            })
            data = {'holdings': holdings, 'sectors': sectors}
            if self.cache_data:
                self.cache.setdefault('holdings', {})[fund_code] = data
                self._save_cache()
            return data
        except Exception as e:
            self._log(f"持仓数据获取失败: {e}")
            return {'holdings': [], 'sectors': []}

class InvestmentStrategy:
    """评分与推荐决策"""
    def __init__(self, market_data: dict, personal_strategy: dict):
        self.market_data = market_data
        self.personal_strategy = personal_strategy
        self.points_log = {}

    def score_fund(self, fund_data: dict, fund_info: dict, manager_data: dict, holdings_data: dict) -> tuple[float, dict]:
        score = 0
        self.points_log = {}
        market_trend = self.market_data.get('trend', 'neutral')
        risk_tolerance = self.personal_strategy.get('risk_tolerance', 'medium')
        horizon = self.personal_strategy.get('horizon', 'long-term')
        
        # 动态权重
        rose_weight = 15 if horizon == 'short-term' else 10
        drawdown_weight = 30 if risk_tolerance == 'low' else 15
        scale_weight = 15 if risk_tolerance == 'low' else 10
        
        # 指标
        rose_3y = fund_info.get('rose(3y)', np.nan)
        rank_r_3y = fund_info.get('rank_r(3y)', np.nan)
        rose_1y = fund_info.get('rose_1y', np.nan)
        rose_6m = fund_info.get('rose_6m', np.nan)
        scale = fund_info.get('scale', np.nan)
        sharpe_ratio = fund_data.get('sharpe_ratio', np.nan)
        max_drawdown = fund_data.get('max_drawdown', np.nan)
        
        # 评分规则
        if pd.notna(rose_3y) and rose_3y > 100: score += 20; self.points_log['3年涨幅 > 100%'] = 20
        elif pd.notna(rose_3y) and rose_3y > 50: score += 10; self.points_log['3年涨幅 > 50%'] = 10
        if pd.notna(rank_r_3y) and rank_r_3y < 0.05: score += 15; self.points_log['3年排名 < 5%'] = 15
        if pd.notna(rose_1y) and rose_1y > 50: score += rose_weight; self.points_log['1年涨幅 > 50%'] = rose_weight
        if pd.notna(rose_6m) and rose_6m > 25: score += rose_weight; self.points_log['6月涨幅 > 25%'] = rose_weight
        if pd.notna(scale) and scale > 10: score += scale_weight; self.points_log['规模 > 10亿'] = scale_weight
        if pd.notna(sharpe_ratio) and sharpe_ratio > 1.0: score += 20; self.points_log['夏普比率 > 1.0'] = 20
        elif pd.notna(sharpe_ratio) and sharpe_ratio > 0.5: score += 10; self.points_log['夏普比率 > 0.5'] = 10
        if pd.notna(max_drawdown) and max_drawdown < 0.2: score += drawdown_weight; self.points_log['最大回撤 < 20%'] = drawdown_weight
        
        # 经理评分
        tenure_years = manager_data.get('tenure_years', np.nan)
        manager_return = manager_data.get('cumulative_return', np.nan)
        if pd.notna(tenure_years) and tenure_years > 3 and pd.notna(manager_return) and manager_return > 20:
            score += 20; self.points_log['基金经理资深且回报高'] = 20
        
        # 持仓集中度
        holdings = holdings_data.get('holdings', [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            top_10_concentration = holdings_df['占净值比例'].iloc[:10].sum() if len(holdings_df) >= 10 else holdings_df['占净值比例'].sum()
            if top_10_concentration > 60:
                score -= 15; self.points_log['前十持仓集中度 > 60%'] = -15
        sectors = holdings_data.get('sectors', [])
        if sectors:
            sectors_df = pd.DataFrame(sectors)
            max_sector = sectors_df['占净值比例'].max()
            if max_sector > 40: score -= 10; self.points_log['单一行业 > 40%'] = -10
        
        # 市场情绪调整
        original_score = score
        if market_trend == 'bullish':
            score *= 1.1
            self.points_log['市场趋势: 牛市'] = score - original_score
        elif market_trend == 'bearish':
            score *= 0.9
            self.points_log['市场趋势: 熊市'] = score - original_score
        
        return score, self.points_log

    def make_decision(self, score: float) -> str:
        if score > 80:
            return "强烈推荐：核心持仓"
        elif score > 60:
            return "推荐持有：卫星持仓"
        elif score > 30:
            return "观望：需进一步观察"
        else:
            return "不推荐：评分较低"

class FundAnalyzer:
    """批量分析基金并生成推荐报告"""
    def __init__(self, cache_data: bool = True):
        self.data_fetcher = FundDataFetcher(cache_data=cache_data)
        self.analysis_report = []

    def _log(self, message: str):
        self.analysis_report.append(message)
        print(f"[分析报告] {message}")

    def _infer_fund_type(self, fund_name: str) -> str:
        if '指数' in fund_name or 'ETF' in fund_name.upper():
            return '指数型'
        if '债券' in fund_name:
            return '债券型'
        if '混合' in fund_name:
            return '混合型'
        if '股票' in fund_name:
            return '股票型'
        return '未知'

    def analyze_multiple_funds(self, merged_csv: str, personal_strategy: dict, code_column: str = 'code', max_funds: int = None):
        try:
            funds_df = pd.read_csv(merged_csv, encoding='gbk')
            self._log(f"导入 {merged_csv}，共 {len(funds_df)} 只基金")
            funds_df[code_column] = funds_df[code_column].astype(str).str.zfill(6)
            funds_df['类型'] = funds_df['name'].apply(self._infer_fund_type)
            fund_info_dict = funds_df.set_index(code_column).to_dict('index')
            fund_codes = funds_df[code_column].unique().tolist()
            if max_funds:
                fund_codes = fund_codes[:max_funds]
                self._log(f"限制分析前 {max_funds} 只基金")
        except Exception as e:
            self._log(f"导入 CSV 失败: {e}")
            return None

        market_data = self.data_fetcher.get_market_sentiment()
        strategy_engine = InvestmentStrategy(market_data, personal_strategy)
        results = []
        
        for code in fund_codes:
            fund_info = fund_info_dict.get(code, {})
            self._log(f"\n分析基金 {code} ({fund_info.get('name', '未知')})")
            fund_data = self.data_fetcher.get_fund_data(code, fund_info.get('name', '未知'))
            manager_data = self.data_fetcher.get_fund_manager_data(code)
            holdings_data = self.data_fetcher.get_fund_holdings_data(code)
            score, points_log = strategy_engine.score_fund(fund_data, fund_info, manager_data, holdings_data)
            decision = strategy_engine.make_decision(score)
            results.append({
                'fund_code': code,
                'fund_name': fund_info.get('name', 'N/A'),
                'rose_3y': fund_info.get('rose(3y)', np.nan),
                'rank_r_3y': fund_info.get('rank_r(3y)', np.nan),
                'rose_1y': fund_info.get('rose_1y', np.nan),
                'rose_6m': fund_info.get('rose_6m', np.nan),
                'scale': fund_info.get('scale', np.nan),
                'sharpe_ratio': fund_data.get('sharpe_ratio', np.nan),
                'max_drawdown': fund_data.get('max_drawdown', np.nan),
                'manager_name': manager_data.get('name', 'N/A'),
                'decision': decision,
                'score': score
            })
            self._log(f"评分详情: {points_log}")
            time.sleep(2)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('fund_analysis_results.csv', encoding='gbk', index=False)
        
        # 生成Markdown报告
        report_lines = [f"# 基金分析报告\n分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n市场趋势: {market_data.get('trend', 'unknown')}\n"]
        report_lines.append("## 推荐基金（Top 10）\n")
        top_funds = results_df.sort_values('score', ascending=False).head(10)
        if not top_funds.empty:
            report_lines.append("| 基金代码 | 基金名称 | 3年涨幅 | 1年涨幅 | 6月涨幅 | 规模(亿) | 夏普比率 | 最大回撤 | 评分 | 推荐 |\n")
            report_lines.append("|----------|----------|---------|---------|---------|----------|----------|----------|------|------|\n")
            for _, row in top_funds.iterrows():
                report_lines.append(
                    f"| {row['fund_code']} | {row['fund_name']} | {row['rose_3y']:.2f}% | {row['rose_1y']:.2f}% | "
                    f"{row['rose_6m']:.2f}% | {row['scale']:.2f} | {row['sharpe_ratio']:.4f} | {row['max_drawdown']:.4f} | {row['score']:.2f} | {row['decision']} |\n"
                )
        else:
            report_lines.append("无推荐基金\n")
        
        report_lines.append("\n## 所有基金分析结果\n")
        report_lines.append("| 基金代码 | 基金名称 | 3年涨幅 | 1年涨幅 | 6月涨幅 | 规模(亿) | 评分 | 推荐 |\n")
        report_lines.append("|----------|----------|---------|---------|---------|----------|------|------|\n")
        for _, row in results_df.iterrows():
            report_lines.append(
                f"| {row['fund_code']} | {row['fund_name']} | {row['rose_3y']:.2f}% | {row['rose_1y']:.2f}% | "
                f"{row['rose_6m']:.2f}% | {row['scale']:.2f} | {row['score']:.2f} | {row['decision']} |\n"
            )
        
        report_path = 'fund_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        self._log(f"分析报告已保存至: {report_path}")
        return results_df

def main():
    # 运行 integrated_fund_screener.py 的 main_scraper 生成 merged_funds.csv
    from integrated_fund_screener import main_scraper
    main_scraper()
    
    # 分析合并数据
    analyzer = FundAnalyzer(cache_data=True)
    personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }
    results_df = analyzer.analyze_multiple_funds('merged_funds.csv', personal_strategy, code_column='code', max_funds=50)

if __name__ == '__main__':
    main()
```
