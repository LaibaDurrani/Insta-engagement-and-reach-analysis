# Quick Start Guide

## Running the Flask Dashboard

1. **Install dependencies** (if not already installed):
   ```bash
   pip install flask pandas numpy plotly
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open in browser**:
   - Navigate to: http://localhost:5000
   - You'll see the onboarding page
   - Select your role and goals
   - Click "Continue" to view your personalized dashboard

## Key Differences from Streamlit

| Feature | Streamlit | Flask |
|---------|-----------|-------|
| **Startup** | `streamlit run app.py` | `python app.py` |
| **Port** | Default 8501 | Default 5000 |
| **Reload** | Auto-reload on file changes | Manual restart (or use debug mode) |
| **State Management** | `st.session_state` | Flask sessions |
| **UI Updates** | Automatic reruns | AJAX/fetch requests |

## Testing the Application

1. **Start the app**: `python app.py`
2. **Visit**: http://localhost:5000
3. **Test onboarding**:
   - Select a user type (e.g., "Instagram Creator")
   - Check some goals
   - Click "Continue"
4. **Test dashboard**:
   - Verify KPIs load
   - Test filters (platform, post type, etc.)
   - Click "Apply Filters"
   - Toggle "Show All Visualizations"
5. **Check insights** at the bottom of the page

## Production Deployment

For production, consider:

```python
# In app.py, change:
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

Or use a production server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Environment Variables

Set a secure secret key for production:

```python
import os
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
```

Then set the environment variable:
```bash
# Windows PowerShell
$env:SECRET_KEY="your-random-secret-key"

# Linux/Mac
export SECRET_KEY="your-random-secret-key"
```
