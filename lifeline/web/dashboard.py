"""
Web Dashboard - Visualize the Living Transformers

A beautiful, real-time web interface to see Lifeline in action!
Watch events, monitor activity, chat with transformers, all in your browser.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Set
from datetime import datetime

try:
    from aiohttp import web
    import aiohttp_cors
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LifelineDashboard:
    """
    Web dashboard for Lifeline

    Provides a beautiful, real-time interface to:
    - See Lifeline's status and activity
    - Watch events as they happen
    - View insights and learning
    - Chat with transformers
    - Visualize code analysis
    """

    def __init__(self, daemon, port: int = 8765):
        self.daemon = daemon
        self.port = port
        self.app = None
        self.runner = None
        self.site = None

        # WebSocket connections for real-time updates
        self.websockets: Set[web.WebSocketResponse] = set()

        # Event tracking for dashboard
        self.recent_events = []
        self.max_events = 100

    async def start(self):
        """
        Start the web dashboard
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - dashboard disabled")
            logger.info("Install aiohttp: pip install aiohttp aiohttp-cors")
            return

        logger.info(f"🌐 Starting web dashboard on port {self.port}...")

        self.app = web.Application()

        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })

        # Setup routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/status', self.handle_status)
        self.app.router.add_get('/api/events', self.handle_events)
        self.app.router.add_get('/api/insights', self.handle_insights)
        self.app.router.add_get('/api/commits', self.handle_commits)
        self.app.router.add_get('/api/memory', self.handle_memory)
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_post('/api/chat', self.handle_chat)

        # Apply CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        # Start the server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()

        logger.info(f"✅ Dashboard running at http://localhost:{self.port}")
        logger.info(f"🎨 Open in your browser to see the transformers LIVE!")

        # Register for daemon events
        self._register_event_handlers()

    async def stop(self):
        """
        Stop the dashboard
        """
        logger.info("🌐 Stopping web dashboard...")

        # Close all websockets
        for ws in self.websockets:
            await ws.close()

        if self.runner:
            await self.runner.cleanup()

        logger.info("✅ Dashboard stopped")

    def _register_event_handlers(self):
        """
        Register to receive daemon events
        """
        # Listen to all events for real-time dashboard updates
        self.daemon.event_loop.on("*", self._on_any_event)

    async def _on_any_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Handle any event for dashboard updates
        """
        # Record event
        event_record = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
        }

        self.recent_events.append(event_record)

        # Keep bounded
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]

        # Broadcast to all connected websockets
        await self._broadcast_event(event_record)

    async def _broadcast_event(self, event: Dict[str, Any]):
        """
        Broadcast event to all websocket clients
        """
        if not self.websockets:
            return

        message = json.dumps({
            "type": "event",
            "event": event,
        })

        # Send to all connected clients
        disconnected = set()
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception:
                disconnected.add(ws)

        # Remove disconnected clients
        self.websockets -= disconnected

    async def handle_index(self, request):
        """
        Serve the dashboard HTML
        """
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type='text/html')

    async def handle_status(self, request):
        """
        API: Get daemon status
        """
        status = self.daemon.get_status()
        return web.json_response(status)

    async def handle_events(self, request):
        """
        API: Get recent events
        """
        count = int(request.query.get('count', 50))
        events = self.recent_events[-count:]

        return web.json_response({
            "events": events,
            "total": len(self.recent_events),
        })

    async def handle_insights(self, request):
        """
        API: Get recent insights
        """
        count = int(request.query.get('count', 20))
        insights = self.daemon.memory.get_recent_insights(count)

        return web.json_response({
            "insights": insights,
            "total": len(self.daemon.memory.insights),
        })

    async def handle_commits(self, request):
        """
        API: Get recent commits
        """
        count = int(request.query.get('count', 20))
        commits = self.daemon.memory.get_recent_commits(count)

        return web.json_response({
            "commits": commits,
            "total": len(self.daemon.memory.commit_history),
        })

    async def handle_memory(self, request):
        """
        API: Get memory statistics
        """
        stats = self.daemon.memory.get_stats()
        return web.json_response(stats)

    async def handle_websocket(self, request):
        """
        WebSocket endpoint for real-time updates
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.websockets.add(ws)
        logger.info(f"🔌 WebSocket connected (total: {len(self.websockets)})")

        # Send current status immediately
        await ws.send_str(json.dumps({
            "type": "status",
            "status": self.daemon.get_status(),
        }))

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle incoming websocket messages
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self.websockets.discard(ws)
            logger.info(f"🔌 WebSocket disconnected (remaining: {len(self.websockets)})")

        return ws

    async def handle_chat(self, request):
        """
        API: Chat with transformers
        """
        try:
            data = await request.json()
            message = data.get('message', '')

            if not message:
                return web.json_response(
                    {"error": "No message provided"},
                    status=400
                )

            # Import voice system
            from lifeline.conversation.voice import TransformerVoice

            # Get or create voice
            if not hasattr(self.daemon, 'voice'):
                self.daemon.voice = TransformerVoice(
                    self.daemon.config.get("conversation", {})
                )
                await self.daemon.voice.initialize()

            # Get response
            if self.daemon.voice.is_ready:
                response = await self.daemon.voice.speak(message)
            else:
                response = "Voice system not initialized. Install transformers: pip install transformers"

            return web.json_response({
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    def _generate_dashboard_html(self) -> str:
        """
        Generate the dashboard HTML
        """
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ Lifeline Dashboard - Living Transformers</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            animation: fadeIn 1s ease-in;
        }

        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: slideUp 0.5s ease-out;
        }

        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }

        .status-indicator.alive {
            background: #4ade80;
            box-shadow: 0 0 10px #4ade80;
        }

        .status-indicator.inactive {
            background: #ef4444;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-value {
            font-weight: bold;
            color: #667eea;
        }

        .event-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .event-item {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            animation: fadeIn 0.3s ease-in;
        }

        .event-type {
            font-weight: bold;
            color: #667eea;
        }

        .event-time {
            font-size: 0.85em;
            color: #888;
        }

        .chat-container {
            grid-column: 1 / -1;
        }

        .chat-messages {
            height: 300px;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            animation: fadeIn 0.3s ease-in;
        }

        .chat-message.user {
            background: #667eea;
            color: white;
            margin-left: 20%;
        }

        .chat-message.bot {
            background: white;
            border: 2px solid #667eea;
            margin-right: 20%;
        }

        .chat-input-container {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #667eea;
            border-radius: 10px;
            font-size: 1em;
        }

        .chat-button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }

        .chat-button:hover {
            background: #764ba2;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .insight-item {
            padding: 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            border-left: 4px solid #fbbf24;
        }

        .commit-item {
            padding: 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            border-left: 4px solid #10b981;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ Lifeline Dashboard ✨</h1>
            <p>Watch the Transformers Come Alive!</p>
        </div>

        <div class="dashboard-grid">
            <!-- Status Card -->
            <div class="card">
                <h2>
                    <span class="status-indicator alive" id="statusIndicator"></span>
                    Status
                </h2>
                <div id="statusMetrics">
                    <div class="metric">
                        <span>Loading...</span>
                    </div>
                </div>
            </div>

            <!-- Events Card -->
            <div class="card">
                <h2>📊 Live Events</h2>
                <div class="event-list" id="eventList">
                    <p style="text-align: center; color: #888;">Waiting for events...</p>
                </div>
            </div>

            <!-- Insights Card -->
            <div class="card">
                <h2>💡 Recent Insights</h2>
                <div class="event-list" id="insightList">
                    <p style="text-align: center; color: #888;">Loading insights...</p>
                </div>
            </div>

            <!-- Chat Card -->
            <div class="card chat-container">
                <h2>💬 Chat with Transformers</h2>
                <div class="chat-messages" id="chatMessages">
                    <div class="chat-message bot">
                        <strong>Lifeline:</strong> Hello! I'm watching your code and ready to help. Ask me anything! ✨
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Type your message...">
                    <button class="chat-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let eventCount = 0;

        // Connect to WebSocket
        function connectWebSocket() {
            ws = new WebSocket('ws://' + window.location.host + '/ws');

            ws.onopen = () => {
                console.log('✅ Connected to Lifeline!');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'status') {
                    updateStatus(data.status);
                } else if (data.type === 'event') {
                    addEvent(data.event);
                }
            };

            ws.onclose = () => {
                console.log('Disconnected. Reconnecting...');
                setTimeout(connectWebSocket, 3000);
            };
        }

        // Update status display
        function updateStatus(status) {
            const html = `
                <div class="metric">
                    <span>Alive</span>
                    <span class="metric-value">${status.alive ? '✅ Yes' : '❌ No'}</span>
                </div>
                <div class="metric">
                    <span>Uptime</span>
                    <span class="metric-value">${status.uptime || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span>Events Processed</span>
                    <span class="metric-value">${status.events_processed || 0}</span>
                </div>
                <div class="metric">
                    <span>Memory Size</span>
                    <span class="metric-value">${formatBytes(status.memory_size)}</span>
                </div>
            `;

            document.getElementById('statusMetrics').innerHTML = html;
        }

        // Add event to the list
        function addEvent(event) {
            eventCount++;
            const eventList = document.getElementById('eventList');

            if (eventCount === 1) {
                eventList.innerHTML = '';
            }

            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            eventDiv.innerHTML = `
                <div class="event-type">${event.type}</div>
                <div class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</div>
            `;

            eventList.insertBefore(eventDiv, eventList.firstChild);

            // Keep only recent events
            while (eventList.children.length > 20) {
                eventList.removeChild(eventList.lastChild);
            }
        }

        // Load insights
        async function loadInsights() {
            try {
                const response = await fetch('/api/insights?count=10');
                const data = await response.json();

                const insightList = document.getElementById('insightList');

                if (data.insights.length === 0) {
                    insightList.innerHTML = '<p style="text-align: center; color: #888;">No insights yet - still learning!</p>';
                    return;
                }

                insightList.innerHTML = data.insights.map(insight => `
                    <div class="insight-item">
                        <div><strong>💡 ${insight.insight}</strong></div>
                        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">
                            ${new Date(insight.timestamp).toLocaleString()}
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load insights:', e);
            }
        }

        // Send chat message
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();

            if (!message) return;

            // Add user message to chat
            addChatMessage(message, 'user');
            input.value = '';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                // Add bot response
                addChatMessage(data.response, 'bot');
            } catch (e) {
                addChatMessage('Sorry, I had trouble understanding. Make sure transformers is installed!', 'bot');
            }
        }

        // Add message to chat
        function addChatMessage(text, sender) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${sender}`;
            messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'Lifeline'}:</strong> ${text}`;

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Handle Enter key in chat
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Format bytes
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }

        // Initialize
        connectWebSocket();
        loadInsights();

        // Refresh insights periodically
        setInterval(loadInsights, 30000);
    </script>
</body>
</html>
        """
