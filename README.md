# YouTube Script Generator

<p align="center">
  <img src="public/fal-logo.svg" alt="fal" height="40">
</p>

Multi-agent AI system for generating YouTube video scripts with voiceover.

**Powered entirely by [fal](https://fal.ai)** - All LLM inference and audio generation runs through fal's infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         fal                             │
├─────────────────────────┬───────────────────────────────┤
│   OpenRouter Endpoint   │      ElevenLabs Endpoint      │
│                         │                               │
│   - GPT-4/5 models      │   - fal-ai/elevenlabs/tts     │
│   - Web search plugin   │   - Multiple voice styles     │
│   - Function calling    │   - Chunked audio processing  │
└─────────────────────────┴───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent System                          │
│                                                         │
│   Producer → Researcher → Writer → Editor → Voice       │
└─────────────────────────────────────────────────────────┘
```

## fal Endpoints

| Service | Endpoint | Usage |
|---------|----------|-------|
| LLM (via OpenRouter) | `fal.run/openrouter/router/openai/v1` | Agent reasoning, coordination, function calling |
| Web Search | `fal.run/openrouter/router` + web plugin | Real-time research |
| Voice Generation | `fal-ai/elevenlabs/tts/eleven-v3` | Script to speech |

Single fal API key. Full stack AI.

## Features

- **7 AI Agents**: Producer, Researcher, Scriptwriter, Editor, Fact-Checker, Creative Director, Voice Artist
- **Real-time collaboration**: Watch agents work together via Server-Sent Events
- **Quality control**: Automatic word count validation and script purity checks
- **TTS integration**: ElevenLabs voiceover generation

## Quick Start

```bash
npm install
cp .env.example .env
npm start
```

Open `http://localhost:3000` and enter your [fal API key](https://fal.ai/dashboard/keys).

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3000` |
| `AGENT_MODEL` | LLM model via OpenRouter | `openai/gpt-4.1` |
| `SEARCH_MODEL` | Model for web search | `openai/gpt-4.1` |
| `ALLOWED_ORIGINS` | CORS whitelist (comma-separated) | `*` |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/project/start` | POST | Start new project |
| `/api/project/respond` | POST | Send user response |
| `/api/project/state/:id` | GET | Get session state |
| `/api/events/:id` | GET | SSE stream |

## License

MIT
