# Creator Insights Dashboard - Flask Application

A modern, minimal social media analytics dashboard built with Flask, featuring personalized insights based on user type.

## Features

- **Onboarding Flow**: Personalized experience based on user role (Instagram Creator, Facebook Page Admin, etc.)
- **Interactive Dashboard**: Dynamic filtering by platform, post type, content niche, audience demographics
- **Visualizations**: 
  - Engagement heatmaps by day and hour
  - Content mix analysis
  - Conversion funnels
  - Performance comparisons
  - Geographic distribution
  - And more!
- **Automated Insights**: Best posting times, top content niches, optimal post types
- **Responsive Design**: Modern, minimal UI that works on all devices

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your dataset is located at:
   ```
   data/processed/social_media_data_with_niches.csv
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── data/
│   └── processed/
│       └── social_media_data_with_niches.csv
├── static/
│   ├── dashboard.css              # Modern minimal styling
│   └── app.js                     # Client-side JavaScript
└── templates/
    ├── onboard.html               # Onboarding page
    └── dashboard.html             # Main dashboard
```

## How It Works

1. **Onboarding**: Users select their role and goals
2. **Dashboard**: Shows personalized visualizations based on user type
3. **Filtering**: Apply filters for platform, post type, niche, age, country
4. **Show All**: Toggle to view all available visualizations
5. **Insights**: Automated recommendations for best posting times and content

## User Types & Default Visualizations

- **Instagram Creator**: Heatmap, Content Mix, Funnel
- **Facebook Page Admin**: Weekend vs Weekday, Age/Gender, Content Mix
- **YouTube Creator**: Best Hour, Funnel, Age Distribution
- **Twitter/X Influencer**: Best Hour, Virality, Niche Performance
- **LinkedIn Professional**: Best Day, Content Mix, Country Map
- **Social Media Analyst**: Platform Comparison, Funnel, Heatmap
- **Small Business Owner**: Best Hour, Best Day, Content Mix
- **Multi-platform Creator**: Heatmap, Platform Comparison, Best Hour
- **Agency / Marketing Team**: Platform Comparison, Funnel, Country Map
- **Beginner Creator**: Best Hour, Best Day, Content Mix

## API Endpoints

- `GET /` - Onboarding page
- `POST /set_preferences` - Save user preferences
- `GET /dashboard` - Main dashboard
- `POST /api/get_data` - Fetch filtered data and charts

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Design**: Modern minimal UI with gradient accents

## Notes

- Session-based user preferences
- Real-time filtering without page reload
- Responsive charts that adapt to screen size
- Clean, professional aesthetic suitable for business use

## Converting from Streamlit

This Flask application replicates all functionality from the original Streamlit app:
- ✅ Onboarding flow with user types and goals
- ✅ Personalized dashboard views
- ✅ All filter options (platform, post type, niche, age, country)
- ✅ "Show All Visualizations" toggle
- ✅ All 12+ chart types
- ✅ Automated insights and recommendations
- ✅ Modern minimal UI design
- ✅ Responsive layout

## Troubleshooting

**Charts not displaying?**
- Ensure Plotly CDN is accessible
- Check browser console for JavaScript errors

**No data showing?**
- Verify dataset exists at `data/processed/social_media_data_with_niches.csv`
- Check that CSV has required columns

**Filters not working?**
- Ensure app.js is loaded correctly
- Check browser console for fetch errors
