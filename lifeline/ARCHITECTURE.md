# 🏗️ Lifeline Architecture

## Overview

Lifeline is built on an **event-driven architecture** where components communicate through a central event bus. This design allows for loose coupling, extensibility, and reactive behavior.

## Core Principles

1. **Event-Driven**: All components communicate via events
2. **Asynchronous**: Built on asyncio for non-blocking operations
3. **Modular**: Each component has a single, clear responsibility
4. **Persistent**: State is preserved across sessions
5. **Safe**: Bounded resources with configurable limits

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LifelineDaemon                         │
│                    (Orchestrator)                           │
└─────────────┬───────────────────────────────────────────────┘
              │
              │ coordinates
              ▼
    ┌─────────────────────────────────────────────────┐
    │              EventLoop                          │
    │          (Nervous System)                       │
    │  - Routes events between components             │
    │  - Pub/sub pattern                              │
    │  - Async event handling                         │
    └───┬─────────────────────────────────────────┬───┘
        │                                         │
        │ emits events                 subscribes │
        ▼                                         ▼
┌───────────────────┐                    ┌───────────────────┐
│   Watchers        │                    │  AI Decision      │
│  (Senses)         │                    │  Engine (Brain)   │
│                   │                    │                   │
│ - FileWatcher     │                    │ - Analyzes events │
│ - GitWatcher      │                    │ - Makes decisions │
└─────────┬─────────┘                    │ - Generates       │
          │                              │   insights        │
          │ monitors                     └─────────┬─────────┘
          ▼                                        │
    ┌──────────┐                                   │ uses
    │  Files   │                                   ▼
    │  Git     │                          ┌─────────────────┐
    └──────────┘                          │ TransformersBrain│
                                          │ (Neural Core)    │
                                          │                  │
                                          │ - Code analysis  │
                                          │ - Pattern recog. │
                                          └──────────────────┘
                │
                │ persists to
                ▼
        ┌──────────────────┐
        │ ContextManager   │
        │ (Memory)         │
        │                  │
        │ - File contexts  │
        │ - Insights       │
        │ - Commit history │
        │ - Patterns       │
        └──────────────────┘
                │
                │ saves to
                ▼
        ┌──────────────────┐
        │  .lifeline/      │
        │  memory.json     │
        └──────────────────┘
```

## Data Flow

### 1. File Change Event Flow

```
File System Change
    │
    ▼
FileWatcher detects change
    │
    ▼
Emit "file:changed" event
    │
    ▼
EventLoop routes event
    │
    ├──▶ ContextManager updates file context
    │
    └──▶ AIDecisionEngine analyzes change
            │
            ▼
        TransformersBrain processes code
            │
            ▼
        Generate insights/alerts
            │
            ▼
        Emit "ai:insight" or "ai:alert" events
            │
            ▼
        ContextManager stores insights
            │
            ▼
        User notification (future)
```

### 2. Git Commit Event Flow

```
Git Commit
    │
    ▼
GitWatcher detects commit
    │
    ▼
Fetch commit details (message, files, author)
    │
    ▼
Emit "git:commit" event
    │
    ▼
EventLoop routes event
    │
    ├──▶ ContextManager records commit
    │
    └──▶ AIDecisionEngine analyzes commit
            │
            ▼
        Analyze message quality
        Analyze scope of changes
        Check for patterns
            │
            ▼
        Generate suggestions if needed
            │
            ▼
        Learn from commit patterns
```

## Component Details

### Core Components

#### LifelineDaemon
- **Role**: Orchestrator
- **Responsibilities**:
  - Initialize all components
  - Coordinate lifecycle
  - Handle shutdown signals
  - Provide status API

#### EventLoop
- **Role**: Nervous System
- **Responsibilities**:
  - Route events between components
  - Maintain event history
  - Execute handlers asynchronously
  - Provide pub/sub interface

#### LifecycleManager
- **Role**: Lifecycle Coordinator
- **Responsibilities**:
  - Manage startup sequence
  - Perform health checks
  - Handle graceful shutdown
  - Monitor system health

### Watcher Components

#### FileWatcher
- **Role**: Eyes (File System Monitoring)
- **Responsibilities**:
  - Monitor file changes
  - Detect creates/modifies/deletes
  - Filter ignored patterns
  - Emit file events

**Strategies**:
- **Primary**: watchdog library (when available)
- **Fallback**: Polling-based monitoring

#### GitWatcher
- **Role**: Memory Tracker
- **Responsibilities**:
  - Monitor git state
  - Detect commits
  - Track branch changes
  - Fetch commit details

**Polling Strategy**:
- Check every N seconds (configurable)
- Compare current state to last known state
- Emit events on changes

### AI Components

#### AIDecisionEngine
- **Role**: Conscious Mind
- **Responsibilities**:
  - Analyze events
  - Make decisions
  - Generate insights
  - Raise alerts
  - Learn patterns

**Decision Points**:
- File changes → Code quality analysis
- New files → Security checks
- Commits → Message & scope analysis
- Patterns → Learning opportunities

#### TransformersBrain
- **Role**: Neural Core
- **Responsibilities**:
  - Code analysis
  - Pattern recognition
  - Intent understanding
  - Suggestion generation

**Analysis Methods**:
- **Current**: Heuristic-based analysis
- **Future**: Full transformer model inference

### Memory Components

#### ContextManager
- **Role**: Long-term Memory
- **Responsibilities**:
  - Persist state to disk
  - Track file contexts
  - Store insights
  - Record commit history
  - Maintain patterns

**Storage**:
- Location: `.lifeline/memory.json`
- Format: JSON
- Bounded: Configurable limits

## Event Types

### File Events
- `file:changed` - File modified
- `file:created` - New file created
- `file:deleted` - File deleted

### Git Events
- `git:commit` - New commit detected
- `git:branch_change` - Branch switched
- `git:merge` - Merge detected

### AI Events
- `ai:insight` - AI generated insight
- `ai:suggestion` - Proactive suggestion
- `ai:alert` - Important alert

## State Management

### In-Memory State
- Event queue
- Event history (bounded)
- Component references
- Current status

### Persistent State
- File contexts
- Insights (last N)
- Commit history (last N)
- Learned patterns
- Session data

**Persistence Strategy**:
- Auto-save every N seconds (configurable)
- Save on shutdown
- Bounded memory to prevent growth

## Extensibility Points

### 1. New Event Types
Add new event types by:
1. Define event in appropriate watcher
2. Emit via EventLoop
3. Register handler in daemon
4. Process in AIDecisionEngine

### 2. New Watchers
Add new watchers by:
1. Implement watcher class
2. Initialize in daemon
3. Start/stop in lifecycle
4. Emit events to EventLoop

### 3. New AI Analyses
Add new analyses by:
1. Implement in TransformersBrain
2. Call from AIDecisionEngine
3. Define decision logic
4. Emit appropriate events

### 4. New Storage
Add new storage by:
1. Implement storage adapter
2. Update ContextManager
3. Add to save/load cycle

## Performance Considerations

### Async Operations
- All I/O is async
- Non-blocking event handling
- Concurrent event processing

### Bounded Resources
- Event history limited
- Memory bounded by config
- File size limits
- Pattern storage limits

### Efficient Polling
- Configurable intervals
- Smart diffing (only changes)
- Ignored pattern filtering

## Security

### Safe Defaults
- Ignore sensitive directories
- Don't analyze large files
- Configurable size limits

### Pattern Detection
- Security vulnerability scanning
- Credential detection
- Injection pattern detection

### Privacy
- Local processing by default
- No external API calls (configurable)
- User data stays on disk

## Future Enhancements

### Scalability
- Multi-repository support
- Distributed deployment
- Shared memory across repos

### Intelligence
- Full transformer model integration
- Fine-tuned code models
- Contextual understanding

### Integration
- IDE plugins
- Web dashboard
- Team collaboration
- CI/CD integration

## Design Philosophy

1. **Simple by Default**: Easy to start, powerful when needed
2. **Safe by Design**: Bounded resources, safe defaults
3. **Observable**: Clear logging, status APIs
4. **Extensible**: Easy to add new capabilities
5. **Local-First**: Privacy and performance
6. **Event-Driven**: Reactive and modular
7. **Persistent**: Never forget context

---

This architecture enables Lifeline to be a living, learning system that grows with your codebase while remaining performant, safe, and extensible.
