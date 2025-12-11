from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
from datetime import datetime
import base64

try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Platform radar chart will use manual normalization.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Helper function to decode Plotly binary data
def decode_plotly_bdata(fig_dict):
    """Decode binary-encoded data in Plotly figures to regular arrays"""
    def decode_value(val):
        """Recursively decode any value that might contain bdata or numpy arrays"""
        if isinstance(val, dict) and 'bdata' in val:
            # Decode base64 binary data
            dtype = val.get('dtype', 'f8')
            bdata = base64.b64decode(val['bdata'])
            
            # Convert based on dtype
            if dtype == 'f8':  # float64
                arr = np.frombuffer(bdata, dtype=np.float64)
            elif dtype == 'f4':  # float32
                arr = np.frombuffer(bdata, dtype=np.float32)
            elif dtype == 'i8':  # int64
                arr = np.frombuffer(bdata, dtype=np.int64)
            elif dtype == 'i4':  # int32
                arr = np.frombuffer(bdata, dtype=np.int32)
            elif dtype == 'i2':  # int16
                arr = np.frombuffer(bdata, dtype=np.int16)
            elif dtype == 'i1':  # int8
                arr = np.frombuffer(bdata, dtype=np.int8)
            else:
                arr = np.frombuffer(bdata, dtype=np.float64)
            
            # Replace NaN with None for JSON serialization
            arr_list = arr.tolist()
            return [None if isinstance(x, (float, np.floating)) and pd.isna(x) else x for x in arr_list]
        elif isinstance(val, np.ndarray):
            arr_list = val.tolist()
            return [None if isinstance(x, (float, np.floating)) and pd.isna(x) else x for x in arr_list]
        elif isinstance(val, (float, np.floating)) and pd.isna(val):
            return None
        elif isinstance(val, dict):
            return {k: decode_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [decode_value(item) for item in val]
        else:
            return val
    
    return decode_value(fig_dict)

# =========================
# Flask Social Media Dashboard
# - Modern minimal UI with onboarding
# - Personalized dashboards per user type
# - Filters, "Show All" view, recommendations
# - Reads data/processed/social_media_data_with_niches.csv
# =========================

USER_TYPES = [
    'Instagram Creator', 'Facebook Page Admin', 'YouTube Creator', 'Twitter/X Influencer',
    'LinkedIn Professional', 'Social Media Analyst', 'Small Business Owner', 'Multi-platform Creator',
    'Agency / Marketing Team', 'Beginner Creator'
]

# -------------------------
# Data Loading
# -------------------------
def load_data(path="data/processed/social_media_data_with_niches.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not load dataset at {path}: {e}")
        return None
    
    # Rename 'co' column to 'Platform' if it exists
    if 'co' in df.columns:
        df.rename(columns={'co': 'Platform'}, inplace=True)
    
    # Use 'True Engagement Rate' if 'Engagement Rate' doesn't exist
    if 'True Engagement Rate' in df.columns and 'Engagement Rate' not in df.columns:
        df['Engagement Rate'] = df['True Engagement Rate']
    
    # Basic cleaning and derived columns
    if 'Hour' in df.columns:
        try:
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce').fillna(0).astype(int)
        except:
            pass
    if 'Weekday' in df.columns:
        df['Weekday'] = df['Weekday'].astype(str)
    
    # Derived engagement rate if not present - convert to percentage (0-100)
    if 'Engagement Rate' not in df.columns:
        if {'Likes','Comments','Shares','Impressions'}.issubset(df.columns):
            df['Engagement Rate'] = ((df['Likes'] + df['Comments'] + df['Shares']) / df['Impressions']) * 100
        else:
            df['Engagement Rate'] = np.nan
    else:
        # If Engagement Rate exists but is in decimal form (0-1), convert to percentage (0-100)
        if df['Engagement Rate'].max() <= 1.0:
            df['Engagement Rate'] = df['Engagement Rate'] * 100
    
    # Normalize content niche
    if 'Content Niche' in df.columns:
        df['Content Niche'] = df['Content Niche'].fillna('General')
    
    return df

# Load data once at startup
df_global = load_data()

def get_aggregates(df):
    aggs = {}
    aggs['total_posts'] = len(df)
    
    # Handle NaN values by replacing with 0
    eng_rate = df['Engagement Rate'].mean()
    aggs['avg_engagement_rate'] = 0 if pd.isna(eng_rate) else eng_rate
    
    if 'Impressions' in df.columns:
        avg_imp = df['Impressions'].mean()
        aggs['avg_impressions'] = 0 if pd.isna(avg_imp) else avg_imp
    else:
        aggs['avg_impressions'] = 0
    
    return aggs

# -------------------------
# Chart Functions
# -------------------------
def heatmap_day_hour(df, metric='Engagement Rate'):
    print(f"DEBUG heatmap_day_hour: Starting. Rows={len(df)}, Metric={metric}")
    if 'Weekday' not in df.columns or 'Hour' not in df.columns or 'Post Type' not in df.columns:
        print("DEBUG heatmap_day_hour: Missing required columns")
        return {}
    
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    
    # Find which post type has highest engagement for each day/hour combination
    best_type_data = []
    for day in weekday_order:
        for hour in range(24):
            day_hour_df = df[(df['Weekday'] == day) & (df['Hour'] == hour)]
            if len(day_hour_df) > 0:
                # Calculate average engagement by post type for this day/hour
                type_avg = day_hour_df.groupby('Post Type')[metric].mean()
                if len(type_avg) > 0 and not pd.isna(type_avg).all():
                    best_type = type_avg.idxmax()
                    avg_engagement = type_avg.max()
                    if not pd.isna(avg_engagement):
                        best_type_data.append({
                            'Weekday': day,
                            'Hour': hour,
                            'Best_Type': best_type,
                            'Engagement': avg_engagement
                        })
    
    if not best_type_data:
        print("DEBUG heatmap_day_hour: No best_type_data")
        return {}
    
    print(f"DEBUG heatmap_day_hour: Created {len(best_type_data)} data points")
    result_df = pd.DataFrame(best_type_data)
    result_df['Weekday'] = pd.Categorical(result_df['Weekday'], categories=weekday_order, ordered=True)
    result_df = result_df.sort_values(['Weekday','Hour'])
    
    # Create pivot for engagement values
    z_pivot = result_df.pivot(index='Weekday', columns='Hour', values='Engagement')
    
    # Create text annotations showing which post type is best
    text_pivot = result_df.pivot(index='Weekday', columns='Hour', values='Best_Type')
    
    # Convert to lists for JSON serialization
    z_values = z_pivot.values.tolist()
    text_values = text_pivot.values.tolist()
    
    # Create heatmap with annotations
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=list(range(24)),
        y=weekday_order,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Best Type: %{customdata}<br>Avg Engagement: %{z:.2f}%<extra></extra>',
        customdata=text_values
    ))
    
    fig.update_layout(
        title=f'{metric} by Day & Hour (Best Performing Content Type)',
        xaxis_title='Hour',
        yaxis_title='Day',
        height=420,
        margin=dict(l=20,r=20,t=60,b=20)
    )
    
    print("DEBUG heatmap_day_hour: Chart created successfully")
    return decode_plotly_bdata(fig.to_dict())

def metrics_by_weekday_heatmap(df):
    """Heatmap showing Likes, Shares, Comments, Impressions by day of week with post type sub-rows"""
    if 'Weekday' not in df.columns or 'Post Type' not in df.columns:
        return {}
    
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    
    # Only include Likes, Comments, and Shares metrics
    metrics = []
    if 'Likes' in df.columns:
        metrics.append('Likes')
    if 'Comments' in df.columns:
        metrics.append('Comments')
    if 'Shares' in df.columns:
        metrics.append('Shares')
    
    print(f"DEBUG: Metrics to display: {metrics}")
    print(f"DEBUG: DataFrame has {len(df)} rows")
    
    if not metrics:
        return {}
    
    # Get unique post types (limit to top 3 most common)
    post_types = df['Post Type'].value_counts().head(3).index.tolist()
    if not post_types:
        return {}
    
    # Build heatmap data grouped by metric
    heatmap_data = []
    y_labels = []
    hover_text = []
    
    for metric in metrics:
        for post_type in post_types:
            row = []
            hover_row = []
            for day in weekday_order:
                # Filter for specific day and post type
                filtered_data = df[(df['Weekday'] == day) & (df['Post Type'] == post_type)]
                
                if len(filtered_data) > 0 and metric in filtered_data.columns:
                    avg_value = filtered_data[metric].mean()
                    count = len(filtered_data)
                    
                    if pd.isna(avg_value):
                        row.append(0)
                        hover_row.append(f'Metric: {metric}<br>Post Type: {post_type}<br>Day: {day}<br>Avg: 0<br>Posts: {count}')
                    else:
                        row.append(avg_value)
                        hover_row.append(f'Metric: {metric}<br>Post Type: {post_type}<br>Day: {day}<br>Avg: {avg_value:.2f}<br>Posts: {count}')
                else:
                    row.append(0)
                    hover_row.append(f'Metric: {metric}<br>Post Type: {post_type}<br>Day: {day}<br>Avg: 0<br>Posts: 0')
            
            print(f"DEBUG: {metric} - {post_type}: {row[:3]}... (sum={sum(row):.2f})")
            heatmap_data.append(row)
            hover_text.append(hover_row)
            
            # Include metric name in the label
            y_labels.append(f'{metric} - {post_type}')
    
    # Normalize each row to percentage of its max value for better color visibility
    # This ensures all metrics show color variation regardless of their absolute scale
    normalized_data = []
    for i, row in enumerate(heatmap_data):
        max_val = max(row) if max(row) > 0 else 1
        normalized_row = [(val / max_val) * 100 for val in row]
        normalized_data.append(normalized_row)
        print(f"DEBUG: Row {i} - Label: {y_labels[i]}, Original: {row[:3]}..., Max: {max_val:.2f}, Normalized: {normalized_row[:3]}...")
    
    # Create figure with normalized data
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data,
        x=weekday_order,
        y=y_labels,
        colorscale='Blues',
        hoverongaps=False,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(title="")
    ))
    
    # No need for annotations anymore since metric is in the y-label
    
    fig.update_layout(
        title='Performance Metrics by Day & Post Type',
        xaxis_title='Day of Week',
        yaxis_title='',
        height=max(350, len(y_labels) * 35),
        margin=dict(l=200,r=20,t=60,b=20),
        yaxis=dict(tickfont=dict(size=11))
    )
    
    return decode_plotly_bdata(fig.to_dict())

def bar_weekend_vs_weekday(df, metric='Engagement Rate'):
    if 'Weekend' not in df:
        return {}
    df2 = df.groupby(['Platform','Weekend'])[metric].mean().reset_index()
    fig = px.bar(df2, x='Platform', y=metric, color='Weekend', barmode='group', title='Weekend vs Weekday Comparison')
    fig.update_layout(height=380, margin=dict(l=20,r=20,t=40,b=20))
    return decode_plotly_bdata(fig.to_dict())

def competitive_niche_analysis(df, metric='Engagement Rate'):
    """Competitive Niche Analysis: bubble chart showing engagement vs shares by niche"""
    print(f"DEBUG competitive_niche_analysis: Starting. Rows={len(df)}, Columns={df.columns.tolist()}")
    
    if 'Content Niche' not in df.columns:
        print("DEBUG competitive_niche_analysis: Content Niche column not found")
        return {}
    
    if 'Shares' not in df.columns:
        print("DEBUG competitive_niche_analysis: Shares column not found")
        return {}
    
    if metric not in df.columns:
        print(f"DEBUG competitive_niche_analysis: Metric '{metric}' not found")
        return {}
    
    # Aggregate by Content Niche
    agg_dict = {
        metric: 'mean',
        'Shares': 'mean'
    }
    
    niche_stats = df.groupby('Content Niche').agg(agg_dict).reset_index()
    niche_stats['Post Count'] = df.groupby('Content Niche').size().values
    niche_stats.columns = ['Content Niche', 'Avg Engagement', 'Avg Shares', 'Post Count']
    
    print(f"DEBUG competitive_niche_analysis: Aggregated {len(niche_stats)} niches")
    print(f"DEBUG competitive_niche_analysis: Sample data:\n{niche_stats.head()}")
    
    # Remove rows with NaN values
    niche_stats = niche_stats.dropna()
    
    if niche_stats.empty:
        print("DEBUG competitive_niche_analysis: No data after dropna")
        return {}
    
    # Create scatter plot with bubble sizes
    # Calculate appropriate sizeref for better bubble scaling
    max_post_count = niche_stats['Post Count'].max()
    sizeref = max_post_count / (15 ** 2)  # Adjust denominator to control max bubble size
    
    fig = px.scatter(
        niche_stats,
        x='Avg Engagement',
        y='Avg Shares',
        size='Post Count',
        color='Content Niche',
        hover_data=['Content Niche', 'Avg Engagement', 'Avg Shares', 'Post Count'],
        title='Competitive Niche Analysis (All Platform)',
        labels={
            'Avg Engagement': 'Avg Engagement Rate',
            'Avg Shares': 'Avg Shares',
            'Post Count': 'Number of Posts'
        }
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    fig.update_traces(marker=dict(
        sizemode='area',
        sizeref=sizeref,
        sizemin=4,
        line=dict(width=2, color='white'),
        opacity=0.7
    ))
    
    print("DEBUG competitive_niche_analysis: Chart created successfully")
    return decode_plotly_bdata(fig.to_dict())

def viral_content_patterns(df):
    """Viral Content Patterns: Treemap showing Platform → Post Type → Niche for top 5% posts"""
    required_cols = {'Platform', 'Post Type', 'Content Niche', 'Share Amplification', 'Engagement Rate'}
    if not required_cols.issubset(df.columns):
        print(f"DEBUG viral_content_patterns: Missing required columns. Have: {df.columns.tolist()}")
        return {}
    
    # Filter for top 5% posts by engagement rate
    threshold = df['Engagement Rate'].quantile(0.95)
    top_posts = df[df['Engagement Rate'] >= threshold].copy()
    
    print(f"DEBUG viral_content_patterns: Threshold={threshold:.2f}, Top posts={len(top_posts)}")
    
    if len(top_posts) == 0:
        print("DEBUG viral_content_patterns: No top posts found")
        return {}
    
    # Aggregate by Platform, Post Type, and Content Niche
    viral_stats = top_posts.groupby(['Platform', 'Post Type', 'Content Niche']).agg({
        'Share Amplification': 'mean',
        'Engagement Rate': 'count'
    }).reset_index()
    viral_stats.columns = ['Platform', 'Post Type', 'Content Niche', 'Avg Share Amplification', 'Post Count']
    
    print(f"DEBUG viral_content_patterns: Aggregated {len(viral_stats)} combinations")
    print(f"DEBUG viral_content_patterns: Sample:\n{viral_stats.head()}")
    
    # Use plotly express treemap with path for simplicity
    viral_stats['Full Path'] = (viral_stats['Platform'] + '/' + 
                                 viral_stats['Post Type'] + '/' + 
                                 viral_stats['Content Niche'])
    
    fig = px.treemap(
        viral_stats,
        path=['Platform', 'Post Type', 'Content Niche'],
        values='Post Count',
        color='Avg Share Amplification',
        color_continuous_scale='Blues',
        title='Viral Content Patterns: Platform → Post Type → Niche (Top 5% Posts)'
    )
    
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=60, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        coloraxis_showscale=False
    )
    
    # Update treemap to use more horizontal space
    fig.update_traces(
        textposition='middle center',
        textinfo='label+value',
        tiling=dict(packing='squarify', pad=4),
        hovertemplate='<b>%{label}</b><br>Post Count: %{value}<br>Avg Share Amplification: %{color:.2f}<extra></extra>',
        textfont=dict(size=11),
        insidetextfont=dict(size=10)
    )
    
    print("DEBUG viral_content_patterns: Chart created successfully")
    return decode_plotly_bdata(fig.to_dict())

def funnel_plot(df):
    platforms = df['Platform'].unique().tolist() if 'Platform' in df else []
    fig = go.Figure()
    for p in platforms:
        sub = df[df['Platform']==p]
        impressions = sub['Impressions'].sum() if 'Impressions' in sub else np.nan
        reach = sub['Reach'].sum() if 'Reach' in sub else np.nan
        engagement = (sub[['Likes','Comments','Shares']].sum().sum() if set(['Likes','Comments','Shares']).issubset(sub.columns) else np.nan)
        values = [impressions, reach, engagement]
        fig.add_trace(go.Funnel(
            name=p, 
            y=['Impressions','Reach','Engagement'], 
            x=values,
            textposition='inside',
            textinfo='value'
        ))
    fig.update_layout(
        title='Impressions → Reach → Engagement Funnel', 
        height=480,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return decode_plotly_bdata(fig.to_dict())

def best_hour_line(df, metric='Engagement Rate'):
    if 'Hour' not in df:
        return {}
    df2 = df.groupby('Hour')[metric].mean().reset_index()
    # Convert to lists to avoid binary encoding
    df2['Hour'] = df2['Hour'].tolist()
    df2[metric] = df2[metric].tolist()
    fig = px.line(df2, x='Hour', y=metric, markers=True, title=f'Average {metric} by Hour')
    fig.update_layout(xaxis=dict(dtick=1), height=380)
    return decode_plotly_bdata(fig.to_dict())

def best_day_bar(df, metric='Engagement Rate'):
    if 'Weekday' not in df:
        return {}
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['Weekday'] = pd.Categorical(df['Weekday'], categories=weekday_order, ordered=True)
    df2 = df.groupby('Weekday', observed=True)[metric].mean().reset_index()
    fig = px.bar(df2, x='Weekday', y=metric, title=f'Average {metric} by Day')
    fig.update_layout(height=360)
    return decode_plotly_bdata(fig.to_dict())

def engagement_by_age(df, metric='Engagement Rate'):
    if 'Audience Age' not in df:
        return {}
    df2 = df.groupby('Audience Age')[metric].mean().reset_index()
    fig = px.scatter(df2, x='Audience Age', y=metric, title=f'{metric} vs Audience Age')
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=360)
    return decode_plotly_bdata(fig.to_dict())

def engagement_by_gender(df, metric='Engagement Rate'):
    if 'Audience Gender' not in df:
        return {}
    df2 = df.groupby('Audience Gender')[metric].mean().reset_index()
    fig = px.bar(df2, x='Audience Gender', y=metric, title=f'{metric} by Audience Gender')
    fig.update_layout(height=320)
    return decode_plotly_bdata(fig.to_dict())

def virality_dist(df):
    if 'Virality Score v2' not in df:
        return {}
    fig = px.histogram(df, x='Virality Score v2', nbins=50, title='Virality Score Distribution')
    fig.update_layout(height=320)
    return decode_plotly_bdata(fig.to_dict())

def niche_performance(df, metric='Engagement Rate'):
    if 'Content Niche' not in df:
        return {}
    df2 = df.groupby('Content Niche')[metric].mean().reset_index().sort_values(metric, ascending=False)
    fig = px.bar(df2, x='Content Niche', y=metric, title=f'Content Niche Performance ({metric})')
    fig.update_layout(height=380)
    return decode_plotly_bdata(fig.to_dict())

def country_choropleth(df, metric='Engagement Rate'):
    if 'Audience Location' not in df:
        return {}
    df2 = df.groupby('Audience Location')[metric].mean().reset_index().rename(columns={'Audience Location':'country'})
    # Use country names for now, as ISO-3 conversion would require a mapping library
    # Custom colorscale using funnel plot colors (blue, coral, green, purple)
    custom_colorscale = [
        [0.0, 'rgb(99, 110, 250)'],      # LinkedIn blue
        [0.33, 'rgb(239, 85, 59)'],      # Instagram coral/red
        [0.66, 'rgb(0, 204, 150)'],      # Twitter green
        [1.0, 'rgb(171, 99, 250)']       # Facebook purple
    ]
    fig = px.choropleth(
        df2, 
        locations='country', 
        locationmode='country names', 
        color=metric, 
        title=f'{metric} by Country',
        color_continuous_scale=custom_colorscale
    )
    fig.update_layout(
        height=500, 
        margin=dict(l=10,r=10,t=40,b=10), 
        showlegend=True,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return decode_plotly_bdata(fig.to_dict())

def platform_comp(df):
    if 'Impressions' in df.columns:
        df2 = df.groupby('Platform').agg({'Impressions':'mean','Engagement Rate':'mean'}).reset_index()
        fig = px.scatter(df2, x='Impressions', y='Engagement Rate', size='Impressions', hover_name='Platform', title='Platform Benchmark')
    else:
        df2 = df.groupby('Platform').agg({'Engagement Rate':'mean'}).reset_index()
        fig = px.scatter(df2, x='Engagement Rate', y='Engagement Rate', hover_name='Platform', title='Platform Benchmark')
    fig.update_layout(height=420)
    return decode_plotly_bdata(fig.to_dict())

def platform_radar_chart(df):
    """Radar chart showing platform performance across key metrics (normalized 0-100) using MinMaxScaler"""
    print(f"DEBUG platform_radar: Starting with {len(df)} rows")
    print(f"DEBUG platform_radar: Available columns: {df.columns.tolist()}")
    
    if 'Platform' not in df.columns:
        print("DEBUG platform_radar: No Platform column found")
        return {}
    
    # Define metrics to aggregate (matching notebook code exactly)
    agg_dict = {}
    metric_labels = []
    
    # Check for True Engagement Rate first, then fallback to Engagement Rate
    if 'True Engagement Rate' in df.columns:
        agg_dict['True Engagement Rate'] = 'median'
        metric_labels.append('True Engagement Rate')
        print("DEBUG platform_radar: Found True Engagement Rate")
    elif 'Engagement Rate' in df.columns:
        agg_dict['Engagement Rate'] = 'median'
        metric_labels.append('True Engagement Rate')
        print("DEBUG platform_radar: Using Engagement Rate as True Engagement Rate")
    
    if 'Interaction Rate' in df.columns:
        agg_dict['Interaction Rate'] = 'median'
        metric_labels.append('Interaction Rate')
        print("DEBUG platform_radar: Found Interaction Rate")
    
    if 'Share Amplification' in df.columns:
        agg_dict['Share Amplification'] = 'median'
        metric_labels.append('Share Amplification')
        print("DEBUG platform_radar: Found Share Amplification")
    
    if 'Exposure Efficiency' in df.columns:
        agg_dict['Exposure Efficiency'] = 'median'
        metric_labels.append('Exposure Efficiency')
        print("DEBUG platform_radar: Found Exposure Efficiency")
    
    if 'Reach' in df.columns:
        agg_dict['Reach'] = 'mean'
        metric_labels.append('Reach')
        print("DEBUG platform_radar: Found Reach")
    
    print(f"DEBUG platform_radar: Total metrics found: {len(agg_dict)}")
    print(f"DEBUG platform_radar: Metrics to aggregate: {list(agg_dict.keys())}")
    
    if not agg_dict:
        print("DEBUG platform_radar: No metrics available")
        return {}
    
    # Aggregate metrics by platform
    try:
        platform_metrics = df.groupby('Platform').agg(agg_dict).round(2)
        print(f"DEBUG platform_radar: Aggregated data shape: {platform_metrics.shape}")
        print(f"DEBUG platform_radar: Platform metrics:\n{platform_metrics}")
    except Exception as e:
        print(f"DEBUG platform_radar: Error aggregating data: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    if len(platform_metrics) == 0:
        print("DEBUG platform_radar: No platform data after aggregation")
        return {}
    
    print(f"DEBUG platform_radar: Platforms: {platform_metrics.index.tolist()}")
    print(f"DEBUG platform_radar: Metrics: {list(platform_metrics.columns)}")
    
    # Normalize metrics to 0-100 scale using MinMaxScaler (matching notebook)
    if HAS_SKLEARN:
        scaler = MinMaxScaler(feature_range=(0, 100))
        platform_metrics_scaled = pd.DataFrame(
            scaler.fit_transform(platform_metrics),
            columns=platform_metrics.columns,
            index=platform_metrics.index
        )
        print("DEBUG platform_radar: Using sklearn MinMaxScaler for normalization")
    else:
        # Manual normalization if sklearn not available
        platform_metrics_scaled = platform_metrics.copy()
        for col in platform_metrics.columns:
            min_val = platform_metrics[col].min()
            max_val = platform_metrics[col].max()
            if max_val > min_val:
                platform_metrics_scaled[col] = ((platform_metrics[col] - min_val) / (max_val - min_val)) * 100
            else:
                platform_metrics_scaled[col] = 50
        print("DEBUG platform_radar: Using manual normalization (sklearn not available)")
    
    # Create the figure
    fig = go.Figure()
    
    # Define colors for each platform to match the reference image exactly
    platform_colors = {
        'Facebook': {'line': 'rgb(138, 123, 221)', 'fill': 'rgba(138, 123, 221, 0.65)'},
        'Instagram': {'line': 'rgb(255, 127, 114)', 'fill': 'rgba(255, 127, 114, 0.65)'},
        'LinkedIn': {'line': 'rgb(92, 219, 211)', 'fill': 'rgba(92, 219, 211, 0.65)'},
        'Twitter': {'line': 'rgb(180, 151, 231)', 'fill': 'rgba(180, 151, 231, 0.65)'}
    }
    
    # Add trace for each platform
    for platform in platform_metrics_scaled.index:
        values = platform_metrics_scaled.loc[platform].values.tolist()
        print(f"DEBUG platform_radar: {platform} scaled values: {values}")
        
        color_config = platform_colors.get(platform, {'line': 'rgb(100, 100, 100)', 'fill': 'rgba(100, 100, 100, 0.65)'})
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels if metric_labels else platform_metrics_scaled.columns.tolist(),
            fill='toself',
            name=platform,
            line=dict(color=color_config['line'], width=2.5),
            fillcolor=color_config['fill'],
            marker=dict(size=7, symbol='circle'),
            opacity=1.0
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='',
                tickfont=dict(size=10),
                gridcolor='rgba(189, 199, 216, 0.4)',
                gridwidth=1
            ),
            angularaxis=dict(
                gridcolor='rgba(189, 199, 216, 0.4)',
                gridwidth=1,
                linecolor='rgba(189, 199, 216, 0.6)'
            ),
            bgcolor='rgba(225, 235, 245, 0.6)'
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(200, 200, 200, 0.3)',
            borderwidth=1
        ),
        title='Platform Performance Across Key Metrics (Normalized 0-100)',
        height=550,
        font=dict(size=11, color='rgb(60, 70, 90)'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    print("DEBUG platform_radar_chart: Chart created successfully")
    return decode_plotly_bdata(fig.to_dict())

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('onboard.html', user_types=USER_TYPES)

@app.route('/set_preferences', methods=['POST'])
def set_preferences():
    data = request.get_json()
    session['selected_user_type'] = data.get('user_type', 'Explorer')
    session['selected_goals'] = data.get('goals', [])
    return jsonify({'success': True})

@app.route('/dashboard')
def dashboard():
    if df_global is None:
        return "Dataset not found. Please ensure data/processed/social_media_data_with_niches.csv exists.", 500
    
    user_type = session.get('selected_user_type', 'Explorer')
    goals = session.get('selected_goals', [])
    
    # Get filter options
    all_countries = sorted(df_global['Audience Location'].dropna().unique().tolist()) if 'Audience Location' in df_global.columns else []
    all_genders = sorted(df_global['Audience Gender'].dropna().unique().tolist()) if 'Audience Gender' in df_global.columns else []
    
    filter_options = {
        'platforms': sorted(df_global['Platform'].unique().tolist()) if 'Platform' in df_global.columns else [],
        'post_types': sorted(df_global['Post Type'].astype(str).unique().tolist()) if 'Post Type' in df_global.columns else [],
        'niches': sorted(df_global['Content Niche'].astype(str).unique().tolist()) if 'Content Niche' in df_global.columns else [],
        'countries': all_countries,
        'genders': all_genders,
        'age_min': int(df_global['Audience Age'].min()) if 'Audience Age' in df_global.columns else 18,
        'age_max': int(df_global['Audience Age'].max()) if 'Audience Age' in df_global.columns else 65,
    }
    
    return render_template('dashboard.html', 
                         user_type=user_type, 
                         goals=goals,
                         filter_options=filter_options)

@app.route('/api/get_data', methods=['POST'])
def get_data():
    if df_global is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    filters = request.get_json()
    df_filtered = df_global.copy()
    
    print(f"DEBUG: Starting with {len(df_filtered)} rows")
    
    # Apply filters
    if filters.get('platforms'):
        df_filtered = df_filtered[df_filtered['Platform'].isin(filters['platforms'])]
        print(f"DEBUG: After platform filter: {len(df_filtered)} rows")
    if filters.get('post_types'):
        df_filtered = df_filtered[df_filtered['Post Type'].isin(filters['post_types'])]
        print(f"DEBUG: After post_type filter: {len(df_filtered)} rows")
    if filters.get('niches'):
        df_filtered = df_filtered[df_filtered['Content Niche'].isin(filters['niches'])]
        print(f"DEBUG: After niche filter: {len(df_filtered)} rows")
    if filters.get('age_range') and 'Audience Age' in df_filtered.columns:
        age_min, age_max = filters['age_range']
        df_filtered = df_filtered[(df_filtered['Audience Age'] >= age_min) & (df_filtered['Audience Age'] <= age_max)]
        print(f"DEBUG: After age filter: {len(df_filtered)} rows")
    if filters.get('genders') and 'Audience Gender' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Audience Gender'].isin(filters['genders'])]
        print(f"DEBUG: After gender filter: {len(df_filtered)} rows")
    if filters.get('countries') and 'Audience Location' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Audience Location'].isin(filters['countries'])]
        print(f"DEBUG: After country filter: {len(df_filtered)} rows")
    
    # Get metrics - use first selected or default to Engagement Rate
    metrics = filters.get('metrics', ['Engagement Rate'])
    metric = metrics[0] if metrics else 'Engagement Rate'
    user_type = session.get('selected_user_type', 'Explorer')
    show_all = filters.get('show_all', False)
    
    # Get aggregates
    aggs = get_aggregates(df_filtered)
    
    # User type to default graphs mapping
    mapping = {
        'Instagram Creator': ['heatmap','content_mix','viral_patterns','funnel','country_map','platform_radar'],
        'Facebook Page Admin': ['weekend','age_gender','content_mix','country_map','platform_radar'],
        'YouTube Creator': ['best_hour','funnel','age_dist','country_map','platform_radar'],
        'Twitter/X Influencer': ['best_hour','virality','viral_patterns','niche_perf','country_map','platform_radar'],
        'LinkedIn Professional': ['best_day','content_mix','country_map','platform_radar'],
        'Social Media Analyst': ['platform_comp','viral_patterns','funnel','heatmap','platform_radar'],
        'Small Business Owner': ['best_hour','best_day','content_mix','country_map','platform_radar'],
        'Multi-platform Creator': ['heatmap','platform_comp','viral_patterns','best_hour','country_map','platform_radar'],
        'Agency / Marketing Team': ['platform_comp','viral_patterns','funnel','country_map','platform_radar'],
        'Beginner Creator': ['best_hour','best_day','content_mix','country_map','platform_radar'],
        'Explorer': ['heatmap','content_mix','viral_patterns','funnel','country_map','platform_radar']
    }
    
    default_graphs = mapping.get(user_type, mapping['Explorer'])
    
    # Generate charts
    charts = {}
    
    try:
        if show_all or 'heatmap' in default_graphs:
            result = heatmap_day_hour(df_filtered, metric)
            if result and len(result) > 0:
                charts['heatmap'] = result
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    
    try:
        if show_all or 'content_mix' in default_graphs:
            result = competitive_niche_analysis(df_filtered, metric)
            if result:
                charts['content_mix'] = result
    except Exception as e:
        print(f"Error generating competitive_niche_analysis: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if show_all or 'funnel' in default_graphs:
            result = funnel_plot(df_filtered)
            if result and len(result) > 0:
                charts['funnel'] = result
    except Exception as e:
        print(f"Error generating funnel: {e}")
    
    try:
        if show_all or 'weekend' in default_graphs:
            result = bar_weekend_vs_weekday(df_filtered, metric)
            if result and len(result) > 0:
                charts['weekend'] = result
    except Exception as e:
        print(f"Error generating weekend: {e}")
    
    try:
        if show_all or 'best_hour' in default_graphs:
            result = best_hour_line(df_filtered, metric)
            if result and len(result) > 0:
                charts['best_hour'] = result
    except Exception as e:
        print(f"Error generating best_hour: {e}")
    
    try:
        if show_all or 'best_day' in default_graphs:
            result = best_day_bar(df_filtered, metric)
            if result and len(result) > 0:
                charts['best_day'] = result
    except Exception as e:
        print(f"Error generating best_day: {e}")
    
    try:
        if show_all or 'age_gender' in default_graphs or 'age_dist' in default_graphs:
            result = engagement_by_age(df_filtered, metric)
            if result and len(result) > 0:
                charts['age'] = result
    except Exception as e:
        print(f"Error generating age: {e}")
    
    try:
        if show_all or 'age_gender' in default_graphs:
            result = engagement_by_gender(df_filtered, metric)
            if result and len(result) > 0:
                charts['gender'] = result
    except Exception as e:
        print(f"Error generating gender: {e}")
    
    try:
        if show_all or 'virality' in default_graphs:
            result = virality_dist(df_filtered)
            if result and len(result) > 0:
                charts['virality'] = result
    except Exception as e:
        print(f"Error generating virality: {e}")
    
    try:
        if show_all or 'niche_perf' in default_graphs:
            result = niche_performance(df_filtered, metric)
            if result and len(result) > 0:
                charts['niche_perf'] = result
    except Exception as e:
        print(f"Error generating niche_perf: {e}")
    
    try:
        if show_all or 'platform_comp' in default_graphs:
            result = platform_comp(df_filtered)
            if result and len(result) > 0:
                charts['platform_comp'] = result
    except Exception as e:
        print(f"Error generating platform_comp: {e}")
    
    try:
        if show_all or 'country_map' in default_graphs:
            result = country_choropleth(df_filtered, metric)
            if result and len(result) > 0:
                charts['country_map'] = result
    except Exception as e:
        print(f"Error generating country_map: {e}")
    
    try:
        if show_all or 'viral_patterns' in default_graphs:
            result = viral_content_patterns(df_filtered)
            if result:
                charts['viral_patterns'] = result
    except Exception as e:
        print(f"Error generating viral_patterns: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if show_all or 'platform_radar' in default_graphs:
            result = platform_radar_chart(df_filtered)
            if result and len(result) > 0:
                charts['platform_radar'] = result
    except Exception as e:
        print(f"Error generating platform_radar: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate insights
    insights = {}
    try:
        by_hour = df_filtered.groupby('Hour')[metric].mean().sort_values(ascending=False).head(3)
        top_hours = ', '.join([str(int(h)) + ':00' for h in by_hour.index])
        insights['best_hours'] = top_hours
    except:
        insights['best_hours'] = 'Not enough data'
    
    try:
        best_niche = df_filtered.groupby('Content Niche')[metric].mean().idxmax()
        insights['best_niche'] = best_niche
    except:
        insights['best_niche'] = 'Not enough data'
    
    try:
        best_type = df_filtered.groupby('Post Type')[metric].mean().idxmax()
        insights['best_type'] = best_type
    except:
        insights['best_type'] = 'Not enough data'
    
    return jsonify({
        'aggregates': {
            'total_posts': f"{aggs['total_posts']:,}",
            'avg_engagement_rate': f"{aggs['avg_engagement_rate']:.2f}%",
            'avg_impressions': f"{aggs['avg_impressions']:.0f}"
        },
        'charts': charts,
        'insights': insights,
        'default_graphs': default_graphs
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

