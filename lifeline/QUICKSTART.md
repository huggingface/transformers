# 🚀 Lifeline Quick Start Guide

Get up and running with Lifeline in 5 minutes!

## Step 1: Navigate to the Lifeline Directory

```bash
cd /path/to/transformers/lifeline
```

## Step 2: Run the Daemon

```bash
# Start Lifeline!
python -m lifeline run
```

You should see:
```
✨ Starting Lifeline daemon...
📍 Repository: /path/to/transformers

🌟 Awakening Lifeline...
✨ Lifeline is now ALIVE
📍 Watching: /path/to/transformers
🧠 Awareness: ACTIVE
💚 Status: Ready to assist
```

## Step 3: In Another Terminal, Make Some Changes

```bash
# Open another terminal and make a change
echo "# Test change" >> test_file.py

# Or make a commit
git add .
git commit -m "Test lifeline"
```

Watch the daemon terminal - you'll see Lifeline react in real-time!

## Step 4: Check Status

```bash
# In another terminal
python -m lifeline status
```

Output:
```
📊 Lifeline Status
==================================================
Alive: True
Repository: /path/to/transformers
Birth time: 2026-01-10T12:34:56.789
Uptime: 0:05:23
Memory size: 1024 bytes
Events processed: 42
Files watching: 0
```

## Step 5: View Memory

```bash
# See what Lifeline has learned
python -m lifeline memory --stats

# View recent insights
python -m lifeline memory --insights 5

# View recent commits
python -m lifeline memory --commits 5
```

## Step 6: Configure (Optional)

```bash
# Create default config
python -m lifeline config --init

# Edit the config
nano .lifeline/config.json

# View current config
python -m lifeline config --show
```

## Common Commands

```bash
# Start daemon
python -m lifeline run

# Get status
python -m lifeline status

# View statistics
python -m lifeline memory --stats

# See recent activity
python -m lifeline memory --insights 10
python -m lifeline memory --commits 10

# Initialize config
python -m lifeline config --init

# View config
python -m lifeline config --show
```

## What to Expect

Lifeline will:
- ✅ Monitor all file changes in real-time
- ✅ Track git commits and branch changes
- ✅ Analyze code for potential issues
- ✅ Generate insights and suggestions
- ✅ Remember everything across sessions

## Next Steps

- Read the [full README](README.md) for details
- Customize your config in `.lifeline/config.json`
- Watch the logs to see Lifeline learning!

**Happy coding with your AI companion!** ✨
