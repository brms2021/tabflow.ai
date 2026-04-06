# ClearLaunch Project Connection

This project is connected to ClearLaunch board `d16d2cba-c430-4b7a-8b2c-27c2b06674b7`.
Agent: claude-tab-ripper

## MCP Server

The ClearLaunch MCP server is configured in `.mcp.json`.
It provides 80+ tools for board management, card CRUD, epics, releases, and AI agent orchestration.

## Workflow

1. `pickup_next` — get highest-priority unassigned card
2. Read card description, acceptance criteria, technical notes
3. Plan before coding — add plan as card comment
4. Implement, run tests after each change
5. `complete_card` — mark done with summary

## Key Commands

- `pickup_next` / `complete_card` — work lifecycle
- `list_cards` / `create_card` / `update_card` — card management
- `get_project_overview` / `get_board` — context
- `search_cards` / `get_blocked_cards` — discovery
- `generate_standup` / `generate_release_notes` — reporting
