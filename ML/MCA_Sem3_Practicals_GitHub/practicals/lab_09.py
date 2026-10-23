"""Lab 09: EDA and interactive COVID-19 and sales demo dashboards.

Run `python lab_09.py`. Requires plotly for interactive HTML.
The bundled data are explicitly SYNTHETIC and do not measure real events.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from generate_labs_09_10_data import make_covid, make_sales

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'lab_09'
OUT.mkdir(parents=True, exist_ok=True)


def _interactive_figures(title, figures, destination):
    """Export a fully self-contained, interactive Plotly dashboard HTML."""
    from plotly.offline import plot
    html = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>' + title + '</title>',
            '<style>body{font:16px system-ui;background:#f4f6fa;color:#17213a;margin:0;}'
            'header{padding:28px 5%;background:#17213a;color:white}'
            'main{padding:24px 5%;display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}'
            '.panel{background:white;border:1px solid #e6e9f0;border-radius:14px;padding:8px;overflow:hidden}'
            '.warning{background:#fff1db;border-radius:6px;padding:10px 5%;color:#6b4100}</style></head><body>',
            '<header><h1>' + title + '</h1><p>MCA201P | Lab Sheet 09</p></header>',
            '<p class="warning"><strong>SYNTHETIC DATA — classroom demonstration only.</strong> '
            'These values are generated, not actual COVID statistics or sales.</p><main>']
    for i, fig in enumerate(figures):
        html.append('<section class="panel">' + plot(fig, output_type='div',
                      include_plotlyjs=True if i == 0 else False) + '</section>')
    html.append('</main></body></html>')
    destination.write_text('\n'.join(html), encoding='utf-8')


def covid_dashboard() -> dict:
    """EDA of daily cumulative counts and a country comparison dashboard."""
    import plotly.express as px
    path = ROOT / 'datasets' / 'lab_09_covid_SYNTHETIC.csv'
    if not path.exists():
        make_covid()
    df = pd.read_csv(path, parse_dates=['date'])
    metrics = ['confirmed', 'recovered', 'deaths', 'people_vaccinated']
    before_nulls = df.isna().sum().to_dict()
    df = df.dropna(subset=['country', 'date']).sort_values(['country', 'date'])
    df[metrics] = df.groupby('country')[metrics].ffill().fillna(0).clip(lower=0)
    df[metrics] = df.groupby('country')[metrics].cummax()
    df = df.drop_duplicates(subset=['country','date'], keep='last')
    for metric in metrics:
        if (df.groupby('country')[metric].diff().dropna() < 0).any():
            raise ValueError('Non-monotonic cumulative data')
    latest = df.sort_values('date').groupby('country', as_index=False).tail(1)
    latest[['country', *metrics]].to_csv(OUT / 'covid_country_summary.csv', index=False)
    daily = df.copy()
    daily['new_cases'] = daily.groupby('country')['confirmed'].diff().fillna(daily['confirmed'])
    daily['new_deaths'] = daily.groupby('country')['deaths'].diff().fillna(daily['deaths'])
    daily.to_csv(OUT / 'covid_cleaned.csv', index=False)
    figs = [px.line(df, x='date', y='confirmed', color='country', title='Cumulative confirmed cases'),
            px.line(df, x='date', y='recovered', color='country', title='Cumulative recoveries'),
            px.line(df, x='date', y='deaths', color='country', title='Cumulative deaths'),
            px.line(df, x='date', y='people_vaccinated', color='country', title='People vaccinated'),
            px.bar(latest, x='country', y='confirmed', color='country', title='Country comparison: cases'),
            px.line(daily, x='date', y='new_cases', color='country', title='New daily confirmed cases')]
    _interactive_figures('COVID-19 Analytics Dashboard (Simulated)', figs,
                         OUT / 'covid_dashboard.html')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.flatten(), metrics):
        sns.lineplot(data=df, x='date', y=metric, hue='country', ax=ax)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=25)
        ax.legend(fontsize=7)
    fig.suptitle('COVID-19 — synthetic demonstration dataset')
    fig.savefig(OUT / 'covid_dashboard_preview.png', dpi=125)
    plt.close(fig)
    report = {'dataset_rows': len(df), 'countries': int(df.country.nunique()),
              'nulls_before_cleaning': before_nulls,
              'total_confirmed_at_end': int(latest.confirmed.sum()),
              'total_deaths_at_end': int(latest.deaths.sum()),
              'note': 'All figures are simulated and must not be reported as real COVID-19 statistics.'}
    (OUT / 'covid_findings.json').write_text(json.dumps(report, indent=2))
    return report


def sales_dashboard() -> dict:
    """Analyze monthly revenue, profit, products, region and customer segments."""
    import plotly.express as px
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    path = ROOT / 'datasets' / 'lab_09_sales_SYNTHETIC.csv'
    if not path.exists():
        make_sales()
    df = pd.read_csv(path, parse_dates=['date'])
    nulls = df.isna().sum().to_dict()
    df = df.dropna(subset=['order_id', 'customer_id', 'date']).copy()
    for col in ['region','product']:
        df[col] = df[col].fillna('Unknown').astype(str)
    for col in ['sales', 'cost','quantity']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    df['profit'] = df.sales - df.cost
    df['month'] = df.date.dt.to_period('M').astype(str)
    df.to_csv(OUT / 'sales_cleaned.csv', index=False)
    monthly = df.groupby('month', as_index=False)[['sales','profit']].sum()
    products = df.groupby('product', as_index=False)[['sales','profit']].sum()
    regions = df.groupby('region', as_index=False)[['sales','profit']].sum()
    reference = df.date.max() + pd.Timedelta(days=1)
    customers = df.groupby('customer_id').agg(recency=('date', lambda x: (reference-x.max()).days),
                   frequency=('order_id','nunique'), monetary=('sales','sum'))
    scaled = StandardScaler().fit_transform(np.log1p(customers[['recency','frequency','monetary']]))
    customers['segment'] = KMeans(n_clusters=3, random_state=42,n_init=10).fit_predict(scaled)
    customers.to_csv(OUT / 'sales_customer_segments.csv')
    monthly.to_csv(OUT / 'sales_monthly_summary.csv',index=False)
    products.to_csv(OUT / 'sales_product_summary.csv',index=False)
    regions.to_csv(OUT / 'sales_region_summary.csv',index=False)
    segments = customers.groupby('segment', as_index=False).agg(
        customers=('frequency','size'), avg_order_frequency=('frequency','mean'),
        avg_customer_sales=('monetary','mean'))
    segments.to_csv(OUT / 'sales_segment_profiles.csv', index=False)
    figs = [px.line(monthly, x='month', y=['sales','profit'], markers=True,
                    title='Monthly revenue and profit'),
            px.bar(products, x='product',y='sales',color='product', title='Product revenue'),
            px.bar(regions,x='region',y='profit',color='region', title='Regional profit'),
            px.pie(products,names='product',values='sales',title='Product revenue share'),
            px.scatter(customers.reset_index(),x='frequency',y='monetary',color='segment',
                       hover_data=['customer_id'],title='Customer segmentation (RFM)'),
            px.bar(segments,x='segment',y='customers',color='segment',title='Customer count by segment')]
    _interactive_figures('Sales Analytics Dashboard (Simulated)', figs,
                         OUT / 'sales_dashboard.html')
    fig, ax = plt.subplots(2, 2,figsize=(12,8),constrained_layout=True)
    monthly.plot(x='month',y=['sales','profit'],ax=ax[0,0],title='Monthly sales and profit')
    products.plot.bar(x='product',y='sales',ax=ax[0,1],legend=False,title='Products')
    regions.plot.bar(x='region',y='profit',ax=ax[1,0],legend=False,title='Regions')
    sns.scatterplot(data=customers,x='frequency',y='monetary',hue='segment',
                    palette='viridis',ax=ax[1,1])
    ax[1,1].set_title('RFM customer groups')
    fig.suptitle('Sales analytics — synthetic demonstration dataset')
    fig.savefig(OUT / 'sales_dashboard_preview.png',dpi=125)
    plt.close(fig)
    report = {'records':len(df),'distinct_customers':int(df.customer_id.nunique()),
              'total_sales_inr':round(float(df.sales.sum()),2),
              'total_profit_inr':round(float(df.profit.sum()),2),
              'top_product_by_revenue':str(products.loc[products.sales.idxmax(),'product']),
              'top_region_by_profit':str(regions.loc[regions.profit.idxmax(),'region']),
              'nulls_before_cleaning': nulls,
              'note':'All figures are simulated business records, not real transactions.'}
    (OUT / 'sales_findings.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    print('COVID EDA:',json.dumps(covid_dashboard(),indent=2))
    print('Sales EDA:',json.dumps(sales_dashboard(),indent=2))
    print('Interactive dashboards: outputs/lab_09/*.html')


if __name__ == '__main__':
    main()
