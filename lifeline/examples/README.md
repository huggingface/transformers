# Lifeline Examples

This directory contains example configurations and usage patterns for Lifeline.

## Files

- **example_config.json**: A complete example configuration with all available options
- More examples coming soon!

## Usage

To use the example config:

```bash
# Copy to your repository
cp examples/example_config.json ../.lifeline/config.json

# Edit as needed
nano ../.lifeline/config.json

# Run Lifeline
python -m lifeline run
```

## Configuration Options

### Log Level
- `DEBUG`: Verbose logging for development
- `INFO`: Normal operation (default)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

### AI Settings
- `model_name`: Transformer model to use (default: "gpt2")
- `use_local`: Use local models vs API (default: true)
- `alert_threshold`: Confidence threshold for alerts (0-1)
- `suggestion_threshold`: Confidence threshold for suggestions (0-1)

### Watcher Settings
- `file_poll_interval`: How often to check files (seconds)
- `git_poll_interval`: How often to check git (seconds)
- `ignored_patterns`: Files/directories to ignore

### Safety Limits
- `max_memory_size`: Maximum memory file size (bytes)
- `max_insights_stored`: Maximum insights to keep
- `max_commits_stored`: Maximum commits to remember
- `auto_save_interval`: How often to save memory (seconds)

### Features
Enable or disable specific features:
- `proactive_suggestions`: Get automatic suggestions
- `security_alerts`: Security vulnerability alerts
- `code_quality_analysis`: Code quality checks
- `commit_analysis`: Analyze commits

## Examples Coming Soon

- Multi-repository setup
- Custom notification handlers
- Integration with CI/CD
- Advanced AI configurations
