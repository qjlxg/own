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
import pytz

class FundAnalyzer:
    """整合基金数据获取、筛选、分析和报告生成的全流程"""
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.risk_free_rate = self._get_risk_free_rate()
        self._web_headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
            ]),
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'http://fund.eastmoney.com/'
        }
        self.failed_funds = []
        self.output_dir = 'data'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _log(self, message: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def _load_cache(self):
        if self.cache_data and os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_data:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def _get_risk_free_rate(self):
        # 假设获取国债收益率作为无风险利率，这里简化为固定值
        return 0.025

    def _download_fund_csv(self, fund_code, start_date='20200101', end_date=datetime.now().strftime('%Y%m%d')):
        """下载基金历史净值并计算收益率"""
        file_path = os.path.join(self.output_dir, f"{fund_code}_fund_history.csv")
        try:
            # akshare可能更稳定，优先使用
            df = ak.fund_etf_hist_em(fund=fund_code, period="每日", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty:
                self._log(f"基金 {fund_code} 历史净值数据为空。")
                return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
            df.columns = ['日期', '单位净值', '累计净值', '日增长率', '申购状态', '赎回状态', '分红送配', '大额赎回']
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df.to_csv(file_path, encoding='utf-8')

            # 计算收益率
            today = df.index[-1]
            one_year_ago = today - timedelta(days=365)
            six_months_ago = today - timedelta(days=180)
            
            nav_1y_ago = df.loc[df.index >= one_year_ago, '单位净值'].iloc[0] if not df.loc[df.index >= one_year_ago, '单位净值'].empty else np.nan
            nav_6m_ago = df.loc[df.index >= six_months_ago, '单位净值'].iloc[0] if not df.loc[df.index >= six_months_ago, '单位净值'].empty else np.nan
            nav_today = df.iloc[-1]['单位净值']

            rose_1y = ((nav_today / nav_1y_ago) - 1) * 100 if pd.notna(nav_1y_ago) else np.nan
            rose_6m = ((nav_today / nav_6m_ago) - 1) * 100 if pd.notna(nav_6m_ago) else np.nan

            self._log(f"基金 {fund_code} 历史净值已下载。")
            return {'csv_filename': file_path, 'rose_1y': rose_1y, 'rose_6m': rose_6m}
        except Exception as e:
            self._log(f"基金 {fund_code} 历史净值下载失败: {e}")
            self.failed_funds.append(fund_code)
            return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}

    def _get_fund_managers(self, fund_code):
        """获取基金经理信息，增强鲁棒性"""
        try:
            url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
            response = requests.get(url, headers=self._web_headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 尝试多种方法寻找基金经理信息
            managers = []
            # 优先寻找表格中的链接
            manager_links = soup.find_all('a', href=re.compile(r'manager/\d+\.html'))
            if manager_links:
                managers = [link.text.strip() for link in manager_links]
            else:
                # 如果没有找到表格，尝试寻找其他地方的文本信息
                manager_info_label = soup.find('label', text=re.compile(r'基金经理'))
                if manager_info_label:
                    manager_text = manager_info_label.find_next_sibling(text=True).strip().replace('：', '')
                    managers = [m.strip() for m in re.split(r'、|，|&nbsp;|&nbsp;', manager_text) if m.strip()]

            if not managers:
                self._log(f"获取基金 {fund_code} 基金经理数据失败: 未找到经理信息")
                return "N/A"
            return ", ".join(managers)
        except Exception as e:
            self._log(f"获取基金 {fund_code} 基金经理数据失败: {e}")
            return "N/A"

    def _get_fund_details(self, fund_code):
        """获取基金规模，增强鲁棒性"""
        try:
            url = f"http://fund.eastmoney.com/{fund_code}.html"
            response = requests.get(url, headers=self._web_headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            scale = 'N/A'
            # 寻找资产规模的标签
            scale_label = soup.find('label', text=re.compile(r'资产规模'))
            if scale_label:
                scale_text = scale_label.find_next_sibling('span').text.strip()
                match = re.search(r'([\d.]+)亿元', scale_text)
                if match:
                    scale = float(match.group(1))

            manager = self._get_fund_managers(fund_code)
            self._log(f"基金 {fund_code} 详情已获取。")
            return {'scale': scale, 'manager': manager}
        except Exception as e:
            self._log(f"获取基金 {fund_code} 详情失败: {e}")
            return {'scale': 'N/A', 'manager': 'N/A'}

    def _analyze_fund_risk_metrics(self, file_path):
        """分析基金风险指标（夏普比率、最大回撤）"""
        try:
            if not os.path.exists(file_path):
                self._log(f"分析基金风险参数失败: 文件 {file_path} 不存在")
                return {'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            
            df = pd.read_csv(file_path, index_col='日期', parse_dates=True)
            df['daily_return'] = df['单位净值'].pct_change()
            
            # 夏普比率
            sharpe_ratio = (df['daily_return'].mean() - self.risk_free_rate/252) / df['daily_return'].std() * np.sqrt(252)
            
            # 最大回撤
            cum_returns = (1 + df['daily_return']).cumprod()
            rolling_max = cum_returns.cummax()
            drawdown = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown}
        except Exception as e:
            self._log(f"分析基金风险参数失败: {e}")
            return {'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            
    def run_analysis(self, fund_codes):
        """主运行函数，整合所有步骤"""
        results_list = []
        for i, fund_code in enumerate(fund_codes, 1):
            self._log(f"[{i}/{len(fund_codes)}] 处理基金 {fund_code}...")
            
            # 1. 下载历史净值并计算收益率
            data = self._download_fund_csv(fund_code)
            
            if data['csv_filename']:
                # 2. 获取基金详情和基金经理
                details = self._get_fund_details(fund_code)
                # 3. 分析风险指标
                risk_metrics = self._analyze_fund_risk_metrics(data['csv_filename'])
                
                # 4. 合并所有数据
                results_list.append({
                    'fund_code': fund_code,
                    'fund_name': 'N/A', # 名称可以后续添加或从其他接口获取
                    'rose_1y': data['rose_1y'],
                    'rose_6m': data['rose_6m'],
                    'scale': details['scale'],
                    'manager': details['manager'],
                    'sharpe_ratio': risk_metrics['sharpe_ratio'],
                    'max_drawdown': risk_metrics['max_drawdown']
                })
            else:
                # 如果下载失败，添加空数据
                results_list.append({
                    'fund_code': fund_code,
                    'fund_name': 'N/A',
                    'rose_1y': np.nan,
                    'rose_6m': np.nan,
                    'scale': 'N/A',
                    'manager': 'N/A',
                    'sharpe_ratio': np.nan,
                    'max_drawdown': np.nan
                })
                
        results_df = pd.DataFrame(results_list)
        results_df.set_index('fund_code', inplace=True)
        results_df.to_csv('fund_analysis_results.csv', encoding='utf-8')

        # 评分和报告生成（这里简化评分逻辑，可根据需要调整）
        results_df['score'] = (results_df['rose_1y'] * 0.4) + (results_df['rose_6m'] * 0.3) + (results_df['sharpe_ratio'] * 100)
        results_df['decision'] = results_df['score'].apply(lambda x: '推荐' if x > 20 else '观望')
        
        # 保存失败基金列表
        if self.failed_funds:
            with open('failed_funds.txt', 'w', encoding='utf-8') as f:
                f.write("以下基金数据获取失败，请手动检查：\n")
                f.write("\n".join(self.failed_funds))
        
        # 生成 Markdown 报告
        self._generate_markdown_report(results_df)

    def _generate_markdown_report(self, results_df):
        """生成 Markdown 格式的报告"""
        report_lines = [f"# 基金分析报告\n分析日期: {datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}\n"]
        
        recommended_df = results_df[results_df['decision'] == '推荐'].sort_values('score', ascending=False)
        report_lines.append("\n## 推荐基金 (按评分排序)\n")
        report_lines.append("| 基金代码 | 1年涨幅 | 6月涨幅 | 规模(亿) | 夏普比率 | 最大回撤 | 评分 | 决策 |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        
        if not recommended_df.empty:
            for _, row in recommended_df.iterrows():
                report_lines.append(
                    f"| {row['fund_code']} | {row['rose_1y']:.2f}% | {row['rose_6m']:.2f}% | {row['scale']:.2f} | {row['sharpe_ratio']:.4f} | {row['max_drawdown']:.4f} | {row['score']:.2f} | {row['decision']} |"
                )
        else:
            report_lines.append("无推荐基金\n")
        
        report_lines.append("\n## 所有基金分析结果\n")
        report_lines.append("| 基金代码 | 1年涨幅 | 6月涨幅 | 规模(亿) | 夏普比率 | 最大回撤 | 评分 | 推荐 |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in results_df.iterrows():
            report_lines.append(
                f"| {row['fund_code']} | {row['rose_1y']:.2f}% | {row['rose_6m']:.2f}% | "
                f"{row['scale']:.2f} | {row['sharpe_ratio']:.4f} | {row['max_drawdown']:.4f} | {row['score']:.2f} | {row['decision']} |"
            )
        
        report_path = 'fund_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        self._log(f"分析报告已保存至: {report_path}")
        
def main():
    # 示例基金代码列表，您可以根据需要修改
    fund_codes = ['001724', '001665', '001676'] 
    analyzer = FundAnalyzer()
    analyzer.run_analysis(fund_codes)

if __name__ == '__main__':
    main()
