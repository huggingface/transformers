# 🌐 Lifeline Web Dashboard Guide

## See Your Transformers Come Alive!

The Lifeline Dashboard provides a beautiful, real-time web interface where you can **visualize** everything happening in your codebase. Watch events flow, see insights generate, chat with transformers, all in your browser!

## 🎨 Overview

The dashboard offers:
- **Real-time event visualization** - See every file change, commit, insight as it happens
- **Live status monitoring** - Track daemon health and activity
- **Interactive charts** - Visualize patterns and trends
- **Web-based chat** - Talk to transformers right in your browser
- **Beautiful UI** - Modern, responsive design

## 🚀 Quick Start

### Start the Dashboard

```bash
# Start dashboard on default port (8765)
python -m lifeline dashboard

# Or specify custom port
python -m lifeline dashboard --port 9000
```

You'll see:
```
🌐 Starting web dashboard...

✨ Lifeline is now ALIVE
📍 Watching: /path/to/transformers
🧠 Awareness: ACTIVE
💚 Status: Ready to assist
🌐 Dashboard: http://localhost:8765
```

### Open in Browser

Navigate to: **http://localhost:8765**

You'll see the beautiful Lifeline interface with real-time updates!

## 📊 Dashboard Features

### 1. Status Card

Shows live daemon status:
- ✅ **Alive Status** - Is Lifeline running?
- ⏱️ **Uptime** - How long has it been running?
- 📈 **Events Processed** - Total event count
- 💾 **Memory Size** - Current memory usage

Updates automatically every second!

### 2. Live Events Panel

Watch events in real-time:
- 📝 File changes
- 📦 Git commits
- 🌿 Branch switches
- 💡 AI insights
- ⚠️ Alerts

Each event shows:
- Event type
- Timestamp
- Relevant details

Events scroll automatically with newest on top!

### 3. Recent Insights

See what the AI has learned:
- 💡 Code patterns discovered
- 📊 Development trends
- ⚡ Performance observations
- 🔍 Quality insights

Refreshes every 30 seconds!

### 4. Interactive Chat

Talk to transformers right in the dashboard:
- Type your message
- Get instant responses
- See conversation history
- Beautiful message bubbles

No CLI needed - pure web interface!

## 🏗️ Architecture

### WebSocket Communication

```
Browser <─────WebSocket─────> Dashboard Server
   │                              │
   │     Real-time Events         │
   │ <─────────────────────────── │
   │                              │
   │     Chat Messages            │
   │ ──────────────────────────> │
   │                              │
   └──────────────────────────────┘
```

### Component Flow

```
Daemon Events
    │
    ▼
Dashboard registers wildcard handler
    │
    ▼
Events broadcast via WebSocket
    │
    ▼
Browser receives and displays
    │
    ▼
UI updates in real-time
```

## ⚙️ Configuration

Enable dashboard in `.lifeline/config.json`:

```json
{
  "web": {
    "enabled": true,
    "port": 8765
  }
}
```

### Port Configuration

Choose your port:
- Default: `8765`
- Custom: Any available port
- Behind proxy: Configure reverse proxy

```bash
# Custom port
python -m lifeline dashboard --port 9000
```

## 🎯 Use Cases

### 1. Real-Time Monitoring

Watch your codebase come alive:
```
- Open dashboard
- Start coding
- Watch file changes appear instantly
- See AI analysis in real-time
- Observe pattern learning
```

### 2. Development Insights

Understand your workflow:
```
- Monitor commit patterns
- See which files change most
- Track testing activity
- Observe development rhythm
```

### 3. Team Collaboration

Share with your team:
```
- Run dashboard on shared server
- Team sees same activity
- Collaborate on insights
- Share AI suggestions
```

### 4. Debugging Sessions

Visual debugging:
```
- Watch events during bug reproduction
- See what triggers issues
- Observe system state changes
- Track down edge cases
```

### 5. Presentations & Demos

Show off your AI companion:
```
- Project dashboard on screen
- Make code changes
- Watch transformers react
- Demonstrate AI insights
- Impress your audience!
```

## 🌟 Advanced Features

### Real-Time WebSocket Updates

Every event in the system broadcasts to the dashboard:
- **Zero polling** - True push-based updates
- **Automatic reconnection** - Resilient to network issues
- **Multiple clients** - Many browsers can watch simultaneously
- **Low latency** - Events appear within milliseconds

### API Endpoints

The dashboard provides REST APIs:

#### GET /api/status
```json
{
  "alive": true,
  "uptime": "1:23:45",
  "events_processed": 1234,
  "memory_size": 524288
}
```

#### GET /api/events?count=50
```json
{
  "events": [...],
  "total": 234
}
```

#### GET /api/insights?count=20
```json
{
  "insights": [...],
  "total": 45
}
```

#### GET /api/memory
```json
{
  "total_events_seen": 1234,
  "files_tracked": 567,
  "insights_stored": 89
}
```

#### POST /api/chat
```json
{
  "message": "How does BERT work?",
  "response": "BERT is a transformer model...",
  "timestamp": "2026-01-11T12:34:56"
}
```

## 🎨 UI Customization

### Themes

The dashboard features:
- Beautiful gradient background
- Card-based layout
- Smooth animations
- Responsive design
- Mobile-friendly

### Responsive Design

Works perfectly on:
- 💻 Desktop (large screens)
- 💻 Laptop (medium screens)
- 📱 Tablet (small screens)
- 📱 Mobile (extra small screens)

### Animations

Smooth, delightful animations:
- Fade in for new elements
- Slide up for cards
- Pulse for status indicator
- Scroll animations for events

## 🔧 Integration Examples

### Run with Full Daemon

```bash
# Start daemon with dashboard enabled
python -m lifeline run
```

Enable in config:
```json
{
  "web": {
    "enabled": true,
    "port": 8765
  }
}
```

### Dashboard Only

```bash
# Run just the dashboard
python -m lifeline dashboard --port 8765
```

### Behind Reverse Proxy

**Nginx Example**:
```nginx
server {
    listen 80;
    server_name lifeline.example.com;

    location / {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 💡 Tips & Tricks

### 1. Multiple Dashboards

Run multiple instances:
```bash
# Terminal 1
python -m lifeline dashboard --port 8765 --repo /project1

# Terminal 2
python -m lifeline dashboard --port 8766 --repo /project2
```

### 2. Share with Team

Make accessible on network:
```bash
# Listen on all interfaces
# (dashboard already does this on 0.0.0.0)
python -m lifeline dashboard

# Access from any machine
# http://<your-ip>:8765
```

### 3. Keep Open While Coding

Best workflow:
1. Start dashboard
2. Open in browser
3. Position browser alongside your editor
4. Watch transformers react as you code!

### 4. Use for Presentations

Impressive demos:
1. Project dashboard on screen
2. Live code transformers
3. Show real-time AI analysis
4. Chat with transformers live!

### 5. Monitor Long-Running Tasks

Track progress:
1. Start long task (build, tests, etc)
2. Watch events on dashboard
3. See insights generate
4. Monitor completion

## 🛠️ Troubleshooting

### Dashboard Won't Start

```
⚠️  aiohttp not available - dashboard disabled
   Install aiohttp: pip install aiohttp aiohttp-cors
```

**Solution**:
```bash
pip install aiohttp aiohttp-cors
```

### Port Already in Use

```
Error: Port 8765 already in use
```

**Solution**:
```bash
# Use different port
python -m lifeline dashboard --port 9000
```

### WebSocket Not Connecting

Check:
1. Dashboard is running
2. Correct URL (http://localhost:8765)
3. No firewall blocking
4. Browser supports WebSocket (all modern browsers do)

### Events Not Appearing

Make sure:
1. Daemon is running
2. Repository has activity
3. WebSocket is connected (check browser console)
4. Events are being generated

## 📊 Performance

### Resource Usage

Very lightweight:
- **CPU**: Minimal (async I/O)
- **Memory**: ~50-100MB
- **Network**: Low bandwidth (only deltas)
- **Browser**: Modern, efficient React-like updates

### Scalability

Handles:
- ✅ 100+ events per second
- ✅ 10+ simultaneous browsers
- ✅ Hours of continuous uptime
- ✅ Large repositories (10k+ files)

## 🔮 Future Enhancements

Coming soon:
- 📊 Charts and graphs
- 🎨 Custom themes
- 📱 Mobile app
- 🔔 Desktop notifications
- 📈 Historical analytics
- 🎥 Screen recording
- 🌍 Multi-repository view
- 👥 Team collaboration features

## 📝 API Reference

### WebSocket Messages

**Server → Client**:
```json
{
  "type": "event",
  "event": {
    "type": "file:changed",
    "data": {...},
    "timestamp": "2026-01-11T12:34:56"
  }
}
```

```json
{
  "type": "status",
  "status": {
    "alive": true,
    "uptime": "1:23:45"
  }
}
```

### REST API

All endpoints return JSON.

**GET endpoints**: No auth required (currently)
**POST endpoints**: JSON body required

**Error responses**:
```json
{
  "error": "Error message here"
}
```

## 🎓 Learning Resources

### Understand the Code

Key files:
- `lifeline/web/dashboard.py` - Main dashboard server
- `lifeline/web/__init__.py` - Web module exports

### Extend the Dashboard

Add new features:
1. Add new API endpoint
2. Add corresponding UI
3. Connect via WebSocket or REST
4. Test and enjoy!

### Build Custom Dashboards

Use the APIs to build your own:
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8765/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};

// Fetch status
const status = await fetch('/api/status').then(r => r.json());
console.log('Status:', status);
```

---

**Now open that dashboard and watch your transformers LIVE!** 🌟

The future of development is here - and it's beautiful, aware, and always watching to help you create amazing things! ✨
