# ELCS MCP Server - Quick Start Guide

## Installation

1. **Navigate to the project directory**:
   ```bash
   cd /home/ty/Repositories/ai_workspace/evolutionary-layering-in-complex-systems-mcp
   ```

2. **Create and activate a virtual environment with uv**:
   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

## Configuration for Claude Desktop

1. **Copy the configuration snippet from `example_mcp_config.json`**:
   ```json
   {
     "mcpServers": {
       "elcs-framework": {
         "command": "uv",
         "args": [
           "--directory",
           "/home/ty/Repositories/ai_workspace/evolutionary-layering-in-complex-systems-mcp",
           "run",
           "python",
           "-m",
           "src.enhanced_elcs_mcp.server"
         ],
         "env": {
           "LOG_LEVEL": "INFO"
         }
       }
     }
   }
   ```

2. **Add to your Claude Desktop configuration**:
   - Open your Claude Desktop configuration file
   - Add the above snippet to the `mcpServers` section
   - Restart Claude Desktop

## Using the ELCS Tools

Once configured, you'll have access to these tools:

### Core Tools
- `create_emergence_network` - Create Dynamic Emergence Networks
- `create_swarm_simulation` - Initialize Multi-Agent Swarm Intelligence
- `run_swarm_simulation` - Execute simulation cycles
- `detect_emergence_patterns` - Analyze emergent behaviors
- `make_collective_decision` - Execute collective intelligence decisions
- `get_framework_status` - Check system status
- `optimize_swarm_parameters` - Optimize swarm performance

### Resources
- `enhanced-elcs://framework-documentation` - Complete documentation
- `enhanced-elcs://simulation-templates` - Ready-to-use templates
- `enhanced-elcs://best-practices` - Best practices guide

### Prompts
The server includes guided prompts for:
- Setting up emergence experiments
- Interpreting simulation results
- Optimizing research workflows

## Quick Example

```python
# 1. Create a network
create_emergence_network("test_net", ["cellular", "social"])

# 2. Create a swarm
create_swarm_simulation("test_swarm", "test_net", agent_count=10)

# 3. Run simulation
run_swarm_simulation(
    "test_swarm", 
    {"complexity": 0.7, "resources": 0.8},
    cycles=5
)

# 4. Check results
detect_emergence_patterns("test_swarm")
```

## Troubleshooting

### If the server doesn't start:
1. Check that you're in the virtual environment
2. Verify all dependencies are installed: `uv pip list`
3. Check logs: The server logs to stderr

### If tools aren't available in Claude:
1. Verify the configuration path is correct
2. Restart Claude Desktop after adding configuration
3. Check Claude Desktop logs for connection errors

## Development

To run the server manually for testing:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the server
python -m src.enhanced_elcs_mcp.server
```

## Support

For more detailed information, use these resources within Claude:
- `enhanced-elcs://framework-documentation`
- `enhanced-elcs://best-practices`

Or check the comprehensive guide at:
`/prompts/elcs_tools_guide.md`
