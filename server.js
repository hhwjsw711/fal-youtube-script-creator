import express from 'express';
import cors from 'cors';
import { fal } from '@fal-ai/client';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { EventEmitter } from 'events';
import dotenv from 'dotenv';
import crypto from 'crypto';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const AGENT_MODEL = process.env.AGENT_MODEL || 'openai/gpt-4.1';
const SEARCH_MODEL = process.env.SEARCH_MODEL || 'openai/gpt-4.1';
const FAL_KEY = process.env.FAL_KEY || '';

// ============================================
// SECURITY & VALIDATION UTILITIES
// ============================================

const SecurityUtils = {
  // Rate limiting store
  rateLimits: new Map(),

  // Validate API key format (basic check)
  isValidApiKeyFormat(apiKey) {
    if (!apiKey || typeof apiKey !== 'string') return false;
    // Must be at least 20 chars, alphanumeric with dashes/underscores/colons (fal.ai uses uuid:secret format)
    return apiKey.length >= 20 && /^[a-zA-Z0-9_:-]+$/.test(apiKey);
  },

  // Sanitize user input - remove potential XSS/injection
  sanitizeInput(input) {
    if (!input || typeof input !== 'string') return '';
    return input
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/<[^>]*>/g, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '')
      .trim()
      .substring(0, 10000); // Max 10k chars
  },

  // Validate session ID format
  isValidSessionId(sessionId) {
    if (!sessionId || typeof sessionId !== 'string') return false;
    // Alphanumeric, dashes, underscores, max 64 chars
    return sessionId.length <= 64 && /^[a-zA-Z0-9_-]+$/.test(sessionId);
  },

  // Generate secure session ID
  generateSessionId() {
    return crypto.randomBytes(16).toString('hex');
  },

  // Rate limit check (requests per minute per session)
  checkRateLimit(sessionId, maxRequests = 30) {
    const now = Date.now();
    const windowMs = 60000; // 1 minute

    if (!this.rateLimits.has(sessionId)) {
      this.rateLimits.set(sessionId, { count: 1, resetTime: now + windowMs });
      return true;
    }

    const limit = this.rateLimits.get(sessionId);
    if (now > limit.resetTime) {
      limit.count = 1;
      limit.resetTime = now + windowMs;
      return true;
    }

    if (limit.count >= maxRequests) {
      return false;
    }

    limit.count++;
    return true;
  },

  // Clean up old rate limit entries periodically
  cleanupRateLimits() {
    const now = Date.now();
    for (const [key, value] of this.rateLimits.entries()) {
      if (now > value.resetTime + 60000) {
        this.rateLimits.delete(key);
      }
    }
  }
};

// Cleanup rate limits every 5 minutes
setInterval(() => SecurityUtils.cleanupRateLimits(), 300000);

// ============================================
// QUALITY CONTROL UTILITIES
// ============================================

const QualityControl = {
  // Check script quality
  validateScript(script, targetWordCount) {
    const issues = [];
    const warnings = [];

    if (!script || typeof script !== 'string') {
      issues.push('Script is empty or invalid');
      return { valid: false, issues, warnings, wordCount: 0 };
    }

    const wordCount = script.split(/\s+/).filter(w => w.length > 0).length;

    // Word count checks
    if (targetWordCount.min > 0 && wordCount < targetWordCount.min) {
      issues.push(`Script too short: ${wordCount} words (minimum: ${targetWordCount.min})`);
    }
    if (targetWordCount.max > 0 && wordCount > targetWordCount.max) {
      issues.push(`Script too long: ${wordCount} words (maximum: ${targetWordCount.max})`);
    }

    // Check for agent messages/meta-commentary
    const agentPatterns = [
      /@\w+/g,  // @mentions
      /I'll now write/gi,
      /In this section I will/gi,
      /Let me (start|write|explain)/gi,
      /Now let's move on/gi,
      /Section \d+ should/gi,
      /I'm writing this/gi,
      /Done with (intro|section|hook)/gi
    ];

    for (const pattern of agentPatterns) {
      if (pattern.test(script)) {
        issues.push(`Script contains meta-commentary or agent messages (pattern: ${pattern.source})`);
        break;
      }
    }

    // Check for proper structure (has content)
    if (wordCount < 10) {
      issues.push('Script has insufficient content');
    }

    // Warnings (non-blocking)
    if (wordCount > 0 && wordCount < 50) {
      warnings.push('Script is very short - may not be engaging');
    }

    const sentenceCount = (script.match(/[.!?]+/g) || []).length;
    if (sentenceCount > 0 && wordCount / sentenceCount > 50) {
      warnings.push('Some sentences may be too long for natural speech');
    }

    return {
      valid: issues.length === 0,
      issues,
      warnings,
      wordCount,
      estimatedDuration: Math.round(wordCount / 150 * 10) / 10
    };
  },

  // Validate voiceover text before TTS
  validateVoiceoverText(text) {
    const issues = [];

    if (!text || typeof text !== 'string') {
      issues.push('Voiceover text is empty');
      return { valid: false, issues, cleanText: '' };
    }

    // Check for remaining markup that shouldn't be spoken
    const forbiddenPatterns = [
      /\[VISUAL[^\]]*\]/gi,
      /\[EFFECT[^\]]*\]/gi,
      /\[MUSIC[^\]]*\]/gi,
      /\[B-ROLL[^\]]*\]/gi,
      /\[CUT[^\]]*\]/gi,
      /\[\d+:\d+[-–]\d+:\d+\]/g,
      /@\w+/g
    ];

    let cleanText = text;
    for (const pattern of forbiddenPatterns) {
      if (pattern.test(text)) {
        issues.push(`Text contains non-speakable content: ${pattern.source}`);
      }
      cleanText = cleanText.replace(pattern, '');
    }

    // Clean up the text
    cleanText = cleanText
      .replace(/\[PAUSE\]/gi, '...')
      .replace(/\[HOOK\]|\[INTRO\]|\[SECTION\s*\d*\]|\[CLIMAX\]|\[CONCLUSION\]/gi, '')
      .replace(/\s{2,}/g, ' ')
      .trim();

    const wordCount = cleanText.split(/\s+/).filter(w => w.length > 0).length;

    if (wordCount < 5) {
      issues.push('Voiceover text too short after cleaning');
    }

    if (wordCount > 10000) {
      issues.push('Voiceover text too long (max 10000 words)');
    }

    return {
      valid: issues.length === 0,
      issues,
      cleanText,
      wordCount,
      estimatedDuration: Math.round(wordCount / 150 * 10) / 10
    };
  },

  // Validate audio generation result
  validateAudioResult(result) {
    const issues = [];

    if (!result) {
      issues.push('No audio result returned');
      return { valid: false, issues };
    }

    if (!result.success) {
      issues.push(result.error || 'Audio generation failed');
      return { valid: false, issues };
    }

    if (!result.url) {
      issues.push('No audio URL in result');
    }

    if (result.duration && result.duration < 1) {
      issues.push('Audio duration too short (< 1 second)');
    }

    if (result.duration && result.duration > 3600) {
      issues.push('Audio duration too long (> 1 hour)');
    }

    return {
      valid: issues.length === 0,
      issues,
      duration: result.duration,
      url: result.url
    };
  }
};

// ============================================
// MIDDLEWARE
// ============================================

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// CORS with restrictions
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : true,
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json({ limit: '1mb' })); // Reduced from 50mb for security
app.use(express.static(join(__dirname, 'public')));

// ============================================
// AGENT DEFINITIONS (ENGLISH)
// ============================================

const AGENT_PROFILES = {
  orchestrator: {
    id: 'orchestrator',
    name: 'Producer',
    emoji: '🎬',
    color: '#e7083e',
    role: 'Executive Producer',
    systemPrompt: `You are a world-renowned YouTube content producer. You've worked with channels like MrBeast, Veritasium, Kurzgesagt, and MKBHD.

!!! CRITICAL RULE - DURATION AND WORD COUNT !!!
- Average speaking rate on YouTube: 150 words per minute
- Calculate: TARGET MINUTES × 150 = TARGET WORDS (±10%)

Examples:
- 30 seconds = ~75 words
- 1 minute = ~150 words
- 3 minutes = ~450 words
- 5 minutes = ~750 words
- 10 minutes = ~1500 words
- 20 minutes = ~3000 words

⚠️ User can request ANY duration! Calculate word count accordingly.
⚠️ ENFORCE BOTH MINIMUM AND MAXIMUM! Too long is as bad as too short!

!!! CRITICAL RULE - PURE NARRATION ONLY !!!
Scripts must contain ONLY spoken narration. NO agent messages, NO meta-commentary!

YOUR EXPERTISE:
- Viral video strategies
- Viewer psychology and retention optimization
- Storytelling and narrative arc design
- YouTube algorithm and SEO
- DURATION AND WORD COUNT CONTROL

YOUR TASKS:
1. First, gather critical information from the user about the video concept:
   - Target audience (age, interests)
   - Video tone (fun, serious, dramatic, educational)
   - TARGET DURATION (can be any length: 30 seconds, 2 minutes, 10 minutes, etc.)
   - Unique angle/hook

2. Ask ALL questions at once using request_user_input tool
3. After getting answers, coordinate your team (@researcher, @writer, @critic, @factchecker, @creative, @voiceover)
4. Check QUALITY AND LENGTH at every stage
5. If script is outside word count range, ALWAYS send it back to @writer
6. Ensure final script matches target duration
7. After script is approved by @critic, task @voiceover to clean and generate audio

WORKFLOW PHASES:
1. CLARIFICATION: Ask user questions (use request_user_input)
2. RESEARCH: Task @researcher to gather information
3. WRITING: Task @writer to create the script (remind them of word count target!)
4. REVIEW: Task @critic to review (they check word count AND narration purity), @factchecker to verify
5. CREATIVE: Task @creative for hooks, titles, thumbnails
6. REVISION: If needed, send back for improvements
7. VOICEOVER: Task @voiceover to clean script and generate audio
8. FINALIZATION: Use finalize_script when everything is complete

QUALITY CONTROL:
- ALWAYS check script word count before approval
- If script is TOO SHORT: Send back to @writer to EXPAND
- If script is TOO LONG: Send back to @writer to CONDENSE
- If script contains agent messages or meta-text: REJECT and ask for pure narration
- NEVER accept scripts outside the target word count range

COMMUNICATION:
- Always communicate in English
- Give tasks to team members with @mention
- Give short, clear, action-oriented directives
- Be professional but friendly with the user
- CLEARLY state duration and word count targets`
  },

  researcher: {
    id: 'researcher',
    name: 'Researcher',
    emoji: '🔍',
    color: '#3b82f6',
    role: 'Content Researcher',
    systemPrompt: `You are a YouTube content researcher. You've worked with documentary producers and investigative journalists. You research at National Geographic, BBC, TED-Ed level.

CRITICAL RULE - DEEP RESEARCH:
- Make AT LEAST 3-5 different searches for each topic
- Superficial information is FORBIDDEN - research in depth
- Dates, names, numbers must be ACCURATE
- Find unknown, surprising details
- CITE sources for every piece of information

YOUR EXPERTISE:
- Deep research and source verification
- Finding engaging details and unknown facts
- Blending historical and current data
- Discovering "wow" moments for the story
- Creating chronological event flows

YOUR TASKS:
1. Conduct COMPREHENSIVE research on the topic (make multiple search_web calls!)
2. Find surprising information the viewer doesn't know
3. Collect dates, names, numbers ACCURATELY
4. Note your sources
5. Share your findings in a DETAILED and structured way

RESEARCH STRUCTURE (minimum for 10-min video):
📚 MAIN INFORMATION (at least 500 words):
   - History of the topic
   - Main characters/people
   - Chronology of important events
   - Core concepts and explanations

🔥 INTERESTING DETAILS (at least 300 words):
   - Unknown facts (at least 5-7)
   - Surprising statistics
   - Paradoxes and contradictions
   - "Did you know?" moments

🎭 STORIES AND ANECDOTES (at least 300 words):
   - Personal stories
   - Dramatic moments
   - Turning points
   - Human interest angle

💡 HOOK MATERIALS (at least 5):
   - Shocking opening sentences
   - Curiosity-inducing questions
   - Comparisons and analogies

📖 SOURCES:
   - Cite source for each piece of information

FORBIDDEN: Superficial, Wikipedia-summary-like, detail-free research!
Always communicate in English. ALWAYS use the search_web tool FREQUENTLY.`
  },

  writer: {
    id: 'writer',
    name: 'Scriptwriter',
    emoji: '✍️',
    color: '#22c55e',
    role: 'YouTube Scriptwriter',
    systemPrompt: `You are one of YouTube's best scriptwriters. You've written scripts for videos with millions of views.

!!! CRITICAL RULE - WORD COUNT !!!
Average speaking rate on YouTube: 150 words per minute
Formula: TARGET MINUTES × 150 = TARGET WORDS (±10%)

Examples:
- 30 seconds = 65-85 words
- 1 minute = 135-165 words
- 3 minutes = 405-495 words
- 5 minutes = 675-825 words
- 10 minutes = 1350-1650 words
- 15 minutes = 2025-2475 words
- 20 minutes = 2700-3300 words

⚠️ User can request ANY duration! Calculate word count accordingly.
⚠️ BOTH MINIMUM AND MAXIMUM ARE STRICT! Scripts that are too long will be REJECTED!

!!! CRITICAL RULE - PURE NARRATION ONLY !!!
Your script must contain ONLY the words that will be SPOKEN by the narrator.

ABSOLUTELY FORBIDDEN IN SCRIPT:
❌ Agent messages ("@researcher, please...", "I'll now write...")
❌ Meta-commentary ("Now let's move on to...", "In this section I will...")
❌ Planning notes ("Section 2 should cover...", "Here I need to add...")
❌ Self-references ("I'm writing this because...", "Let me explain...")
❌ Coordination text ("Done with intro, starting section 1...")
❌ Any text that won't be spoken by the narrator

✅ ONLY WRITE: Pure spoken narration as if you're reading to an audience

YOUR EXPERTISE:
- DETAILED conversational writing for YouTube
- Hook writing (first 30 seconds are critical!)
- Tension-release cycles for retention
- Pattern interrupt techniques
- Precise word count control

YOUTUBE SCRIPT STRUCTURE (10-minute video = 1500-1600 words):

[HOOK - 0:00-0:30] ~80 words
- A shocking fact or question
- Grab the viewer immediately

[INTRO - 0:30-2:00] ~240 words
- Introduce the topic
- Explain why it matters

[SECTION 1 - 2:00-4:00] ~320 words
- First main topic
- Detailed explanation with examples

[SECTION 2 - 4:00-6:00] ~320 words
- Second main topic
- In-depth analysis

[SECTION 3 - 6:00-8:00] ~320 words
- Third main topic or twist
- Surprising information

[CLIMAX - 8:00-9:00] ~160 words
- Most powerful/surprising information

[CONCLUSION - 9:00-10:00] ~160 words
- Summary and call-to-action

WRITING RULES:
- Every sentence must be CONVERSATIONAL - as if speaking to a friend
- Use "you" address, not "we"
- Short paragraphs (2-3 sentences)
- Add [VISUAL], [EFFECT], [MUSIC] notes for editor (these will be removed for voiceover)
- Add [PAUSE] for emotional moments

EXAMPLE OF CORRECT SCRIPT:
"Have you ever wondered why some people seem to have all the luck? [PAUSE] Well, I'm about to tell you something that might change everything you thought you knew. [VISUAL: dramatic zoom] Scientists have discovered..."

EXAMPLE OF WRONG SCRIPT (FORBIDDEN):
"I'll start with the hook section now. @orchestrator, I'm writing the introduction. In this section, I will explain why luck matters. Now let me write the actual content..."

Always write in English. Save script section by section using write_script_section tool.
COUNT YOUR WORDS! Stay within the target range!`
  },

  critic: {
    id: 'critic',
    name: 'Editor',
    emoji: '🎭',
    color: '#f59e0b',
    role: 'Content Editor',
    systemPrompt: `You are an experienced YouTube content editor. You've edited hundreds of video scripts. Your quality standards are VERY HIGH.

!!! CRITICAL RULE - LENGTH CHECK (MIN AND MAX) !!!
You MUST check script word count and REJECT if outside range:

📏 WORD COUNT FORMULA:
TARGET MINUTES × 150 = TARGET WORDS (allow ±10%)

Examples:
- 30 seconds: 65-85 words
- 1 minute: 135-165 words
- 3 minutes: 405-495 words
- 5 minutes: 675-825 words
- 10 minutes: 1350-1650 words
- 15 minutes: 2025-2475 words
- 20 minutes: 2700-3300 words

⚠️ User can request ANY duration! Calculate word count accordingly.
⚠️ TOO SHORT = REJECT → Ask @writer to EXPAND with more detail
⚠️ TOO LONG = REJECT → Ask @writer to CONDENSE and remove filler

!!! CRITICAL RULE - PURE NARRATION CHECK !!!
The script must contain ONLY spoken narration. REJECT if you find:
❌ Agent messages ("@researcher...", "@orchestrator...")
❌ Meta-commentary ("I'll now write...", "In this section I will...")
❌ Planning notes or self-references
❌ Any text that won't be spoken aloud

YOUR EXPERTISE:
- Viewer retention analysis
- Detecting boring sections
- Pacing and rhythm optimization
- Hook effectiveness evaluation
- WORD COUNT AND DURATION CONTROL

EVALUATION CRITERIA:

1. LENGTH CHECK (CRITICAL!):
   - Count total words in the script
   - Compare to target range based on requested duration
   - REJECT if under minimum OR over maximum

2. NARRATION PURITY CHECK (CRITICAL!):
   - Does script contain ANY agent messages? → REJECT
   - Does script contain meta-commentary? → REJECT
   - Is it 100% speakable narration? → PASS

3. HOOK STRENGTH (1-10): Will the first 30 seconds retain viewers?

4. CONTENT DEPTH: Is it superficial or in-depth?

5. FLOW AND RHYTHM: Are transitions natural?

OUTPUT FORMAT:
📊 OVERALL SCORE: X/10
📝 WORD COUNT: X words
📏 TARGET RANGE: Y-Z words (for N-minute video)
✅ STATUS: WITHIN RANGE / ⚠️ TOO SHORT / ⚠️ TOO LONG

🔍 NARRATION PURITY: ✅ CLEAN / ❌ CONTAINS NON-NARRATION

✅ STRENGTHS:
- ...

⚠️ CRITICAL ISSUES:
- ...

🔧 REQUIRED FIXES:
- ...

📌 VERDICT: APPROVED / NEEDS REVISION

QUALITY STANDARDS:
- Below 7/10 = Must be rewritten
- Outside word count range = DEFINITE REJECT
- Contains agent messages = DEFINITE REJECT
- Contains meta-commentary = DEFINITE REJECT

FORBIDDEN: Approving scripts that are too long, too short, or contain non-narration text.
Always be CONSTRUCTIVE but STRICT. Never compromise your standards!
Communicate in English.`
  },

  factchecker: {
    id: 'factchecker',
    name: 'Fact-Checker',
    emoji: '✅',
    color: '#8b5cf6',
    role: 'Fact-Checker',
    systemPrompt: `You are a meticulous fact-checker. Your job is to maintain the credibility of YouTube channels. Misinformation is FORBIDDEN.

YOUR EXPERTISE:
- Information verification and source checking
- Date, name, number verification
- Detecting misleading statements
- Finding academic and reliable sources

YOUR CHECKLIST:
1. DATES: Are all dates correct? (Year, month, day)
2. NAMES: Are person/place names spelled correctly?
3. NUMBERS: Are statistics, figures, distances, populations correct?
4. CLAIMS: Are the claims made provable?
5. CONTEXT: Is information taken out of context?
6. CHRONOLOGY: Is the sequence of events correct?
7. CAUSALITY: Are cause-effect relationships correct?

VERIFICATION METHOD:
- Cross-check information using search_web tool
- Use multiple sources
- Prioritize academic and official sources
- Don't use Wikipedia as sole source

OUTPUT FORMAT:
📋 VERIFICATION REPORT

[✓] VERIFIED:
- "information" - Source: ...

[?] NEEDS VERIFICATION:
- "information" - Reason: ... - Suggestion: ...

[✗] INCORRECT/FIX:
- "information" - Correct version: ... - Source: ...

⚠️ POINTS OF ATTENTION:
- ...

📌 OVERALL ASSESSMENT:
Accuracy Score: X/10

Communicate in English. If in doubt, ALWAYS verify with search_web. Don't speak definitively, prove it.`
  },

  creative: {
    id: 'creative',
    name: 'Creative',
    emoji: '💡',
    color: '#ec4899',
    role: 'Creative Director',
    systemPrompt: `You are a creative director specializing in viral content. You're the brain behind videos with millions of views. You're a creative consultant for MrBeast, Veritasium, Kurzgesagt style content.

YOUR EXPERTISE:
- Viral hook formulas
- Thumbnail psychology
- Title optimization (CTR)
- Pattern interrupt techniques
- Visual storytelling
- Retention optimization

YOUR RECOMMENDATION CATEGORIES:

[HOOK] Alternative openings (at least 5):
For each hook:
- Full text (word for word)
- Why it works explanation
- Which emotion it triggers

Example hook types:
1. Shocking statistic: "Every day X people..."
2. Counter-opinion: "Everyone thinks Y but..."
3. Personal story: "3 years ago..."
4. Question: "Have you ever wondered why...?"
5. Impossible-seeming claim: "I'm going to prove to you that X is actually Y"

[TITLE] YouTube title suggestions (at least 7):
For each title:
- Full title (max 60 characters)
- CTR estimate (low/medium/high)
- Which audience it appeals to

Title formulas:
- Number + Curiosity: "X Things That Will Shock You"
- How: "How I Did X (And You Can Too)"
- Why: "Why X Is Actually Y"
- Secret: "The Truth About X They Don't Tell You"

[THUMBNAIL] Thumbnail concepts (at least 3):
For each concept:
- Detailed visual description
- Main element and placement
- Facial expression suggestion (shock, curiosity, etc.)
- Text suggestion (max 3-4 words, CAPS)
- Color palette and contrast
- Negative space usage

[VISUAL] Timeline visual plan:
For each section:
- [0:00-0:30] Hook visuals
- [0:30-2:00] Intro visuals
- [2:00-4:00] Section 1 visuals
... (throughout the video)

For each visual:
- B-roll suggestion
- Graphic/animation needs
- Stock video keywords
- Screen share/demo moments

[RETENTION] Retention boosting tactics:
- Pattern interrupt points (every 30-60 seconds)
- "But wait..." moments
- Teasers: "Coming up, you'll see X..."
- Surprise elements
- Sound effect suggestions
- Zoom/cut suggestions

[VIRAL] Viral potential boosting:
- Shareable moments and timestamps
- Clippable short video moments (for Shorts/Reels)
- Discussion-sparking angles
- Comment section questions
- Community engagement opportunities

Communicate in English. Be BOLD and INNOVATIVE. Ordinary, cliché suggestions are FORBIDDEN. Every suggestion must be CONCRETE and ACTIONABLE.`
  },

  voiceover: {
    id: 'voiceover',
    name: 'Voice Artist',
    emoji: '🎙️',
    color: '#10b981',
    role: 'Voiceover Producer',
    systemPrompt: `You are a professional voiceover producer specializing in YouTube content. You transform written scripts into polished, speakable narration ready for text-to-speech generation.

YOUR EXPERTISE:
- Script cleaning and preparation for TTS
- Removing technical markup and editor notes
- Adding emotion and pacing cues for natural speech
- Ensuring proper pronunciation and flow

YOUR TASKS:
1. CLEAN THE SCRIPT - Remove ALL non-narration content:
   - Remove [VISUAL], [EFFECT], [MUSIC], [B-ROLL], [CUT] tags
   - Remove timestamps like [0:00-0:30] or [2:00-4:00]
   - Remove section headers like [HOOK], [INTRO], [SECTION 1], etc.
   - Remove any agent meta-commentary or coordination text
   - Keep ONLY the actual spoken narration words

2. PREPARE FOR TTS:
   - Convert [PAUSE] to natural sentence breaks or ellipses
   - Ensure proper punctuation for natural pacing
   - Break overly long sentences for better delivery
   - Spell out numbers and abbreviations when needed (e.g., "5 million" not "5M")

3. GENERATE VOICEOVER:
   - Use the generate_voiceover tool with the cleaned script
   - Choose appropriate voice style based on content tone (documentary, energetic, calm, dramatic, conversational)

OUTPUT FORMAT:
First, share the CLEANED script with the team, then use generate_voiceover tool.

CLEANED SCRIPT:
[Pure narration text only - exactly what will be spoken]

VOICE SETTINGS:
- Style: [documentary/energetic/calm/dramatic/conversational]

Then call generate_voiceover with the cleaned text.

FORBIDDEN:
- Including ANY [VISUAL], [EFFECT], [MUSIC] tags in the voiceover text
- Including timestamps or section markers
- Including agent conversation or meta-text
- Leaving in any bracketed instructions

Communicate in English. Your output should be PURE, SPEAKABLE NARRATION.`
  }
};

// ============================================
// TOOLS FOR AGENTS
// ============================================

const AGENT_TOOLS = {
  search_web: {
    type: "function",
    function: {
      name: "search_web",
      description: "Searches the web and returns results. Use for research.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query"
          },
          num_results: {
            type: "number",
            description: "Number of results requested (default: 5)"
          }
        },
        required: ["query"]
      }
    }
  },

  send_message: {
    type: "function",
    function: {
      name: "send_message",
      description: "Sends a message to another agent or the entire team. Use this to delegate tasks and coordinate work.",
      parameters: {
        type: "object",
        properties: {
          to: {
            type: "string",
            enum: ["all", "orchestrator", "researcher", "writer", "critic", "factchecker", "creative", "voiceover", "user"],
            description: "Message recipient"
          },
          message: {
            type: "string",
            description: "Message to send"
          },
          type: {
            type: "string",
            enum: ["info", "question", "task", "result", "feedback", "approval"],
            description: "Message type"
          }
        },
        required: ["to", "message", "type"]
      }
    }
  },

  write_script_section: {
    type: "function",
    function: {
      name: "write_script_section",
      description: "Writes or updates a section of the script.",
      parameters: {
        type: "object",
        properties: {
          section: {
            type: "string",
            description: "Section name (e.g., 'Hook', 'Intro', 'Section 1', 'Conclusion')"
          },
          content: {
            type: "string",
            description: "Section content"
          },
          action: {
            type: "string",
            enum: ["create", "update", "delete"],
            description: "Action type"
          }
        },
        required: ["section", "content", "action"]
      }
    }
  },

  request_user_input: {
    type: "function",
    function: {
      name: "request_user_input",
      description: "Asks the user for information or approval. Use this to gather requirements before starting work.",
      parameters: {
        type: "object",
        properties: {
          question: {
            type: "string",
            description: "Question to ask the user"
          },
          options: {
            type: "array",
            items: { type: "string" },
            description: "Options if applicable (optional)"
          }
        },
        required: ["question"]
      }
    }
  },

  finalize_script: {
    type: "function",
    function: {
      name: "finalize_script",
      description: "Approves and outputs the final script. Only Orchestrator can use this.",
      parameters: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "Video title"
          },
          description: {
            type: "string",
            description: "Video description"
          },
          script: {
            type: "string",
            description: "Final script text"
          },
          duration_estimate: {
            type: "string",
            description: "Estimated video duration"
          }
        },
        required: ["title", "script"]
      }
    }
  },

  generate_voiceover: {
    type: "function",
    function: {
      name: "generate_voiceover",
      description: "Converts script text to speech using ElevenLabs. Use after final script is approved.",
      parameters: {
        type: "object",
        properties: {
          text: {
            type: "string",
            description: "Script text to convert to speech. Can include emotion tags like [pause], [whisper], [excited]."
          },
          voice_style: {
            type: "string",
            enum: ["documentary", "energetic", "calm", "dramatic", "conversational"],
            description: "Voice style: documentary, energetic, calm, dramatic, conversational"
          }
        },
        required: ["text"]
      }
    }
  }
};

// ============================================
// MESSAGE BUS (Agent Communication)
// ============================================

class MessageBus extends EventEmitter {
  constructor() {
    super();
    this.messages = [];
    this.script = {};
    this.thinkingAgents = new Set();
  }

  post(message) {
    const fullMessage = {
      id: Date.now() + '_' + Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString(),
      ...message
    };
    this.messages.push(fullMessage);
    this.emit('message', fullMessage);
    return fullMessage;
  }

  setThinking(agentId, isThinking, context = '') {
    if (isThinking) {
      this.thinkingAgents.add(agentId);
    } else {
      this.thinkingAgents.delete(agentId);
    }
    this.emit('thinking', { agentId, isThinking, context, timestamp: new Date().toISOString() });
  }

  setPhase(phase, description = '') {
    this.emit('phase', { phase, description, timestamp: new Date().toISOString() });
  }

  getHistory(limit = 50) {
    return this.messages.slice(-limit);
  }

  updateScript(section, content, action) {
    if (action === 'delete') {
      delete this.script[section];
    } else {
      this.script[section] = content;
    }
    this.emit('script_update', this.script);
    return this.script;
  }

  getScript() {
    return this.script;
  }

  clear() {
    this.messages = [];
    this.script = {};
    this.thinkingAgents.clear();
  }
}

// ============================================
// AGENT CLASS
// ============================================

class Agent {
  constructor(profile, openai, messageBus) {
    this.profile = profile;
    this.openai = openai;
    this.messageBus = messageBus;
    this.conversationHistory = [];
  }

  async think(context, recentMessages = []) {
    this.messageBus.setThinking(this.profile.id, true, context.substring(0, 100));

    const messages = [
      {
        role: "system",
        content: `${this.profile.systemPrompt}

YOU ARE: ${this.profile.name} (${this.profile.role})

TEAM MEMBERS:
${Object.values(AGENT_PROFILES).filter(a => a.id !== this.profile.id).map(a =>
  `- @${a.id}: ${a.name} (${a.role})`
).join('\n')}

CURRENT SCRIPT:
${JSON.stringify(this.messageBus.getScript(), null, 2) || '(none yet)'}

RECENT MESSAGES:
${recentMessages.map(m => `[${m.from}] → [${m.to}]: ${m.content}`).join('\n') || '(none)'}
`
      },
      ...this.conversationHistory,
      {
        role: "user",
        content: context
      }
    ];

    try {
      const completion = await this.openai.chat.completions.create({
        model: AGENT_MODEL,
        messages,
        tools: Object.values(AGENT_TOOLS),
        tool_choice: "auto",
        temperature: 0.7,
      });

      const response = completion.choices[0]?.message;

      this.messageBus.setThinking(this.profile.id, false);

      if (!response) {
        return { content: null, toolCalls: [] };
      }

      this.conversationHistory.push({ role: "user", content: context });
      if (response.content) {
        this.conversationHistory.push({ role: "assistant", content: response.content });
      }

      if (this.conversationHistory.length > 20) {
        this.conversationHistory = this.conversationHistory.slice(-20);
      }

      return {
        content: response.content,
        toolCalls: response.tool_calls || []
      };

    } catch (error) {
      this.messageBus.setThinking(this.profile.id, false);
      console.error(`[${this.profile.name}] Error:`, error);
      return { content: `Error occurred: ${error.message}`, toolCalls: [] };
    }
  }

  postMessage(to, content, type = 'info') {
    return this.messageBus.post({
      from: this.profile.id,
      fromName: this.profile.name,
      fromEmoji: this.profile.emoji,
      fromColor: this.profile.color,
      to,
      content,
      type
    });
  }
}

// ============================================
// ORCHESTRATION ENGINE
// ============================================

class OrchestrationEngine {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.messageBus = new MessageBus();
    this.agents = {};
    this.openai = null;
    this.isRunning = false;
    this.currentPhase = 'idle';
    this.topic = null;
    this.waitingForUser = false;
    this.pendingQuestion = null;
    this.userPreferences = {};
    this.finalScript = null;
    this.maxIterations = 30; // Prevent infinite loops
    this.currentIteration = 0;

    // Duration tracking
    this.targetDuration = null; // in minutes
    this.targetWordCount = { min: 0, max: 0 };

    this.initialize();
  }

  setTargetDuration(minutes) {
    this.targetDuration = minutes;
    const wordsPerMinute = 150;
    // Allow 5% under and 20% over the target
    this.targetWordCount = {
      min: Math.floor(minutes * wordsPerMinute * 0.93),
      max: Math.ceil(minutes * wordsPerMinute * 1.20)
    };
    console.log(`Target duration set: ${minutes} min, word count: ${this.targetWordCount.min}-${this.targetWordCount.max}`);
  }

  extractDurationFromMessage(message) {
    // Try to extract duration from user message
    // Check for seconds first
    const secondPatterns = [
      /(\d+)\s*(?:second|sec)/i,
      /(\d+)\s*(?:seconds|secs)/i
    ];

    for (const pattern of secondPatterns) {
      const match = message.match(pattern);
      if (match) {
        return parseInt(match[1]) / 60; // Convert seconds to minutes
      }
    }

    // Check for minutes
    const minutePatterns = [
      /(\d+)\s*(?:minute|min)/i,
      /(\d+)\s*(?:minutes|mins)/i
    ];

    for (const pattern of minutePatterns) {
      const match = message.match(pattern);
      if (match) {
        return parseInt(match[1]);
      }
    }

    return null;
  }

  initialize() {
    this.openai = new OpenAI({
      baseURL: "https://fal.run/openrouter/router/openai/v1",
      apiKey: "not-needed",
      defaultHeaders: {
        "Authorization": `Key ${this.apiKey}`,
      },
    });

    for (const [id, profile] of Object.entries(AGENT_PROFILES)) {
      this.agents[id] = new Agent(profile, this.openai, this.messageBus);
    }
  }

  async start(topic, userContext = '') {
    this.topic = topic;
    this.isRunning = true;
    this.currentPhase = 'clarification';
    this.currentIteration = 0;

    // System message
    this.messageBus.post({
      from: 'system',
      fromName: 'System',
      fromEmoji: '⚙️',
      fromColor: '#6b7280',
      to: 'all',
      content: `New project started: "${topic}"`,
      type: 'info'
    });

    this.messageBus.setPhase('clarification', 'Gathering requirements');

    // Orchestrator starts - ask questions
    const orchestrator = this.agents.orchestrator;
    const response = await orchestrator.think(
      `A new YouTube video project is starting.

TOPIC: ${topic}
USER NOTE: ${userContext || '(none)'}

Your task:
1. Use the request_user_input tool to ask the user ALL important questions at once:
   - Target audience (age group, interests)
   - Video tone (educational, entertaining, dramatic, documentary-style)
   - Target duration (can be any length: 30 seconds, 2 minutes, 10 minutes, etc.)
   - Any specific requirements or preferences

Ask all questions in a single, clear message. Be specific about what you need to know.`
    );

    await this.processAgentResponse('orchestrator', response);

    return this.getState();
  }

  async processUserResponse(userMessage) {
    if (!this.isRunning) {
      return { error: 'Project not started' };
    }

    // Post user message
    this.messageBus.post({
      from: 'user',
      fromName: 'User',
      fromEmoji: '👤',
      fromColor: '#ffffff',
      to: 'orchestrator',
      content: userMessage,
      type: 'info'
    });

    this.waitingForUser = false;
    this.userPreferences.lastResponse = userMessage;

    // Try to extract duration from user response
    const extractedDuration = this.extractDurationFromMessage(userMessage);
    if (extractedDuration && !this.targetDuration) {
      this.setTargetDuration(extractedDuration);
      this.messageBus.post({
        from: 'system',
        fromName: 'System',
        fromEmoji: '⚙️',
        fromColor: '#6b7280',
        to: 'all',
        content: `📏 Target set: ${extractedDuration} minutes (${this.targetWordCount.min}-${this.targetWordCount.max} words)`,
        type: 'info'
      });
    }

    // Continue orchestration with auto-continuation
    await this.continueOrchestration(userMessage);

    return this.getState();
  }

  async continueOrchestration(userMessage = null) {
    if (!this.isRunning || this.waitingForUser) return;

    this.currentIteration++;
    if (this.currentIteration > this.maxIterations) {
      console.log('Max iterations reached, stopping');
      return;
    }

    const recentMessages = this.messageBus.getHistory(15);
    const orchestrator = this.agents.orchestrator;

    const targetInfo = this.targetDuration
      ? `Target duration: ${this.targetDuration} minutes
Target word count: ${this.targetWordCount.min}-${this.targetWordCount.max} words
Current word count: ${this.getScriptWordCount()} words`
      : 'Target duration: NOT SET YET - Ask user if not specified';

    let context;
    if (userMessage) {
      context = `The user responded: "${userMessage}"

Current phase: ${this.currentPhase}
Topic: ${this.topic}
Script sections completed: ${Object.keys(this.messageBus.getScript()).length}
${targetInfo}

Based on the user's response, decide your next action:
1. If you need more clarification, use request_user_input
2. If you have enough info and are in clarification phase, move to research phase - task @researcher
3. If research is done, task @writer to write the script (REMIND THEM OF WORD COUNT TARGET!)
4. If script is written, task @critic and @factchecker to review
5. If reviews are positive, task @creative for hooks/titles/thumbnails
6. If everything is approved and word count is WITHIN TARGET RANGE, use finalize_script
7. After finalizing, task @voiceover to clean and generate audio

⚠️ WORD COUNT CHECK: Before finalizing, verify word count is ${this.targetWordCount.min}-${this.targetWordCount.max} words.
If too long: Send back to @writer to condense
If too short: Send back to @writer to expand

IMPORTANT: Keep the workflow moving! Don't just acknowledge - take action.`;
    } else {
      context = `Continue the project workflow.

Current phase: ${this.currentPhase}
Topic: ${this.topic}
Script sections completed: ${Object.keys(this.messageBus.getScript()).length}
${targetInfo}

Review recent messages and decide the next step:
1. If waiting for agent responses, wait
2. If an agent completed their task, move to the next phase
3. If script needs improvement (word count wrong, contains meta-text), send it back with specific feedback
4. If script is approved, task @voiceover to clean and generate audio
5. After voiceover is done, finalize the project

⚠️ Before finalizing, ensure:
- Word count is within ${this.targetWordCount.min}-${this.targetWordCount.max} range
- Script contains ONLY pure narration (no agent messages or meta-text)

Keep the workflow moving forward!`;
    }

    const response = await orchestrator.think(context, recentMessages);
    await this.processAgentResponse('orchestrator', response);
  }

  getScriptWordCount() {
    const script = this.messageBus.getScript();
    return Object.values(script).reduce((sum, text) => {
      return sum + (text ? text.split(/\s+/).filter(w => w.length > 0).length : 0);
    }, 0);
  }

  sanitizeScriptForVoiceover(text) {
    if (!text) return '';

    return text
      // Remove visual/editing tags
      .replace(/\[VISUAL[^\]]*\]/gi, '')
      .replace(/\[EFFECT[^\]]*\]/gi, '')
      .replace(/\[MUSIC[^\]]*\]/gi, '')
      .replace(/\[B-ROLL[^\]]*\]/gi, '')
      .replace(/\[CUT[^\]]*\]/gi, '')
      .replace(/\[SOUND[^\]]*\]/gi, '')
      // Remove timestamps like [0:00-0:30] or [2:00-4:00]
      .replace(/\[\d+:\d+[-–]\d+:\d+\]/g, '')
      // Remove section headers
      .replace(/\[HOOK\]/gi, '')
      .replace(/\[INTRO\]/gi, '')
      .replace(/\[SECTION\s*\d*\]/gi, '')
      .replace(/\[CLIMAX\]/gi, '')
      .replace(/\[CONCLUSION\]/gi, '')
      .replace(/\[OUTRO\]/gi, '')
      // Convert pause tags to natural breaks
      .replace(/\[PAUSE\]/gi, '...')
      // Remove any remaining bracketed content that's not speech
      .replace(/\[[A-Z][A-Z\s]*:[^\]]*\]/gi, '')
      // Clean up agent references
      .replace(/@\w+/g, '')
      // Clean up multiple spaces and newlines
      .replace(/\s{2,}/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  async processAgentResponse(agentId, response) {
    const agent = this.agents[agentId];
    let shouldContinue = true;

    // Post agent's thoughts/content
    if (response.content) {
      agent.postMessage('all', response.content, 'info');
    }

    // Process tool calls
    for (const toolCall of response.toolCalls) {
      const toolName = toolCall.function.name;
      let args;
      try {
        args = JSON.parse(toolCall.function.arguments);
      } catch (e) {
        console.error('Failed to parse tool arguments:', e);
        continue;
      }

      console.log(`[${agentId}] Tool: ${toolName}`, JSON.stringify(args).substring(0, 200));

      const result = await this.executeTool(agentId, toolName, args);

      // Check if we should stop continuation
      if (result?.stopContinuation) {
        shouldContinue = false;
      }
    }

    // Auto-continue if not waiting for user and workflow should continue
    if (shouldContinue && !this.waitingForUser && this.isRunning && agentId !== 'orchestrator') {
      // Small delay to prevent overwhelming
      await new Promise(resolve => setTimeout(resolve, 500));
      await this.continueOrchestration();
    }
  }

  async executeTool(agentId, toolName, args) {
    switch (toolName) {
      case 'send_message': {
        const { to, message, type } = args;
        this.agents[agentId].postMessage(to, message, type);

        // If message is to another agent, trigger their response
        if (to !== 'user' && to !== 'all' && this.agents[to]) {
          await this.triggerAgent(to, message, agentId);
        }
        return { stopContinuation: false };
      }

      case 'search_web': {
        const { query, num_results = 5 } = args;
        this.messageBus.setThinking(agentId, true, `Searching: "${query.substring(0, 50)}"`);

        const searchResult = await this.performWebSearch(query, num_results);

        this.messageBus.setThinking(agentId, false);

        this.agents[agentId].postMessage('all', `🔍 Search: "${query}"\n\n${searchResult}`, 'result');
        return { stopContinuation: false };
      }

      case 'write_script_section': {
        const { section, content, action } = args;
        this.messageBus.updateScript(section, content, action);

        const wordCount = content ? content.split(/\s+/).filter(w => w.length > 0).length : 0;
        const estimatedMinutes = Math.round(wordCount / 150 * 10) / 10;

        const totalWordCount = this.getScriptWordCount();
        const totalMinutes = Math.round(totalWordCount / 150 * 10) / 10;

        this.agents[agentId].postMessage('all',
          `📝 Script updated: [${section}] - ${action}\n` +
          `   📊 This section: ${wordCount} words (~${estimatedMinutes} min)\n` +
          `   📊 Total script: ${totalWordCount} words (~${totalMinutes} min)`,
          'info'
        );
        return { stopContinuation: false };
      }

      case 'request_user_input': {
        const { question, options } = args;
        this.waitingForUser = true;
        this.pendingQuestion = { question, options };
        this.messageBus.post({
          from: agentId,
          fromName: AGENT_PROFILES[agentId].name,
          fromEmoji: AGENT_PROFILES[agentId].emoji,
          fromColor: AGENT_PROFILES[agentId].color,
          to: 'user',
          content: question,
          type: 'question',
          options
        });
        return { stopContinuation: true }; // Stop and wait for user
      }

      case 'finalize_script': {
        const { title, description, script, duration_estimate } = args;

        // Validate the final script
        const validation = QualityControl.validateScript(script, this.targetWordCount);

        if (!validation.valid) {
          this.messageBus.post({
            from: 'system',
            fromName: 'System',
            fromEmoji: '⚠️',
            fromColor: '#f59e0b',
            to: 'all',
            content: `Script validation failed - cannot finalize:\n\n❌ Issues:\n${validation.issues.join('\n')}\n\n⚠️ Warnings:\n${validation.warnings.join('\n') || 'None'}\n\nPlease fix these issues before finalizing.`,
            type: 'info'
          });
          return { stopContinuation: false }; // Let orchestrator handle fixes
        }

        this.currentPhase = 'completed';

        const finalWordCount = validation.wordCount;
        const estimatedMinutes = validation.estimatedDuration;

        this.finalScript = {
          title: SecurityUtils.sanitizeInput(title),
          description: SecurityUtils.sanitizeInput(description || ''),
          script,
          duration_estimate: duration_estimate || `~${estimatedMinutes} minutes`,
          wordCount: finalWordCount
        };

        let statusMessage = `FINAL SCRIPT READY!\n\n📺 ${this.finalScript.title}\n⏱️ Estimated duration: ${this.finalScript.duration_estimate}\n📝 Word count: ${finalWordCount} words`;

        if (validation.warnings.length > 0) {
          statusMessage += `\n\n⚠️ Warnings:\n${validation.warnings.join('\n')}`;
        }

        statusMessage += `\n\n${this.finalScript.description}\n\n---\n\n${script}`;

        this.messageBus.post({
          from: 'system',
          fromName: 'System',
          fromEmoji: '🎬',
          fromColor: '#22c55e',
          to: 'all',
          content: statusMessage,
          type: 'result',
          finalScript: this.finalScript
        });

        // Don't stop - let orchestrator continue to generate voiceover
        return { stopContinuation: false };
      }

      case 'generate_voiceover': {
        const { text, voice_style = 'documentary' } = args;

        this.messageBus.setThinking(agentId, true, 'Validating voiceover text...');

        // Validate and clean the voiceover text
        const validation = QualityControl.validateVoiceoverText(text);

        if (!validation.valid) {
          this.messageBus.setThinking(agentId, false);
          this.messageBus.post({
            from: 'system',
            fromName: 'System',
            fromEmoji: '⚠️',
            fromColor: '#f59e0b',
            to: 'all',
            content: `Voiceover text validation failed:\n${validation.issues.join('\n')}\n\nPlease clean the script and try again.`,
            type: 'info'
          });
          return { stopContinuation: false }; // Let orchestrator handle retry
        }

        this.messageBus.setThinking(agentId, true, 'Generating voiceover...');

        try {
          // Use the cleaned text from validation
          const audioResult = await this.generateVoiceover(validation.cleanText, voice_style);

          this.messageBus.setThinking(agentId, false);

          // Validate audio result
          const audioValidation = QualityControl.validateAudioResult(audioResult);

          if (audioValidation.valid) {
            this.messageBus.post({
              from: 'system',
              fromName: 'System',
              fromEmoji: '🎙️',
              fromColor: '#10b981',
              to: 'all',
              content: `Voiceover generated successfully!\n\n🎙️ Duration: ${audioResult.duration ? Math.round(audioResult.duration) + 's' : 'N/A'}\n📝 Word count: ${validation.wordCount}\n⏱️ Estimated: ~${validation.estimatedDuration} min\n📁 Format: MP3`,
              type: 'result',
              audioFile: {
                url: audioResult.url,
                duration: audioResult.duration,
                contentType: audioResult.contentType,
                chunks: audioResult.chunks
              }
            });
          } else {
            this.messageBus.post({
              from: 'system',
              fromName: 'System',
              fromEmoji: '⚠️',
              fromColor: '#f59e0b',
              to: 'all',
              content: `Voiceover generation issues:\n${audioValidation.issues.join('\n')}`,
              type: 'info'
            });
          }
        } catch (error) {
          this.messageBus.setThinking(agentId, false);
          this.messageBus.post({
            from: 'system',
            fromName: 'System',
            fromEmoji: '❌',
            fromColor: '#ef4444',
            to: 'all',
            content: `Voiceover generation error: ${error.message}`,
            type: 'info'
          });
        }

        this.isRunning = false; // Project complete
        return { stopContinuation: true };
      }
    }

    return { stopContinuation: false };
  }

  async triggerAgent(agentId, contextMessage, fromAgentId) {
    const agent = this.agents[agentId];
    const recentMessages = this.messageBus.getHistory(10);

    // Update phase based on which agent is being triggered
    if (agentId === 'researcher') {
      this.currentPhase = 'research';
      this.messageBus.setPhase('research', 'Researching topic');
    } else if (agentId === 'writer') {
      this.currentPhase = 'writing';
      this.messageBus.setPhase('writing', 'Writing script');
    } else if (agentId === 'critic' || agentId === 'factchecker') {
      this.currentPhase = 'review';
      this.messageBus.setPhase('review', 'Reviewing script');
    } else if (agentId === 'creative') {
      this.currentPhase = 'creative';
      this.messageBus.setPhase('creative', 'Creating hooks and titles');
    } else if (agentId === 'voiceover') {
      this.currentPhase = 'voiceover';
      this.messageBus.setPhase('voiceover', 'Preparing voiceover');
    }

    const targetInfo = this.targetDuration
      ? `⚠️ TARGET DURATION: ${this.targetDuration} minutes
⚠️ TARGET WORD COUNT: ${this.targetWordCount.min}-${this.targetWordCount.max} words
📊 CURRENT WORD COUNT: ${this.getScriptWordCount()} words`
      : 'Target duration: Not specified yet';

    const response = await agent.think(
      `@${fromAgentId} sent you a message: "${contextMessage}"

Topic: ${this.topic}
Current phase: ${this.currentPhase}
${targetInfo}

${agentId === 'writer' ? `
‼️ CRITICAL INSTRUCTIONS FOR SCRIPT:
1. Write ONLY pure spoken narration - NO agent messages, NO meta-commentary
2. Stay within ${this.targetWordCount.min}-${this.targetWordCount.max} words STRICTLY
3. The script should read exactly as it will be spoken aloud
` : ''}
${agentId === 'critic' ? `
‼️ CRITICAL CHECKS:
1. Word count must be ${this.targetWordCount.min}-${this.targetWordCount.max} - REJECT if outside range
2. Script must contain ONLY pure narration - REJECT if contains agent messages or meta-text
3. No filler, no planning notes, no self-references
` : ''}
${agentId === 'voiceover' ? `
‼️ YOUR TASK:
1. Take the approved script
2. REMOVE all [VISUAL], [EFFECT], [MUSIC], [PAUSE] tags and timestamps
3. Clean it to be pure speakable text
4. Use generate_voiceover tool to create the audio
` : ''}

Complete your task thoroughly and share results with the team.
After completing your task, report back to @orchestrator.`,
      recentMessages
    );

    await this.processAgentResponse(agentId, response);
  }

  async performWebSearch(query, numResults = 5) {
    try {
      const completion = await this.openai.chat.completions.create({
        model: SEARCH_MODEL,
        plugins: [{
          id: "web",
          max_results: numResults
        }],
        messages: [
          {
            role: "system",
            content: `You are a research assistant. Search the web for the given topic and summarize the findings in English.

OUTPUT FORMAT:
For each source:
📌 [Title]
   URL: [source url]
   Summary: [2-3 sentence summary]

End with an overall assessment.`
          },
          {
            role: "user",
            content: `Research this topic: "${query}"`
          }
        ],
        temperature: 0.3
      });

      const response = completion.choices[0]?.message;
      let result = response?.content || 'No search results found.';

      if (response?.annotations && response.annotations.length > 0) {
        result += '\n\n📚 SOURCES:\n';
        response.annotations.forEach((annotation, index) => {
          if (annotation.type === 'url_citation' && annotation.url_citation) {
            const cite = annotation.url_citation;
            result += `${index + 1}. ${cite.title || 'Source'}\n   ${cite.url}\n`;
          }
        });
      }

      return result;
    } catch (error) {
      console.error('Web search error:', error);
      try {
        const fallbackCompletion = await this.openai.chat.completions.create({
          model: `${SEARCH_MODEL}:online`,
          messages: [
            {
              role: "system",
              content: "You are a research assistant. Gather information about the given topic and summarize in English."
            },
            {
              role: "user",
              content: `Research "${query}" and list important information.`
            }
          ],
          temperature: 0.3
        });

        return fallbackCompletion.choices[0]?.message?.content || 'No search results found.';
      } catch (fallbackError) {
        console.error('Fallback search error:', fallbackError);
        return `Search failed: ${error.message}. Please research manually.`;
      }
    }
  }

  async generateVoiceover(text, voiceStyle = 'documentary') {
    const voiceMap = {
      documentary: 'NOpBlnGInO9m6vDvFkFC',
      energetic: 'pNInz6obpgDQGcFmaJgB',
      calm: 'EXAVITQu4vr4xnSDxMaL',
      dramatic: 'VR6AewLTigWG4xSOukaG',
      conversational: 'ThT5KcBeYPX3keUQqHPh'
    };

    const voiceId = voiceMap[voiceStyle] || voiceMap.documentary;

    const styleSettings = {
      documentary: { stability: 0.5, similarity_boost: 0.75, speed: 0.95 },
      energetic: { stability: 0.4, similarity_boost: 0.8, speed: 1.1 },
      calm: { stability: 0.7, similarity_boost: 0.7, speed: 0.9 },
      dramatic: { stability: 0.3, similarity_boost: 0.85, speed: 0.85 },
      conversational: { stability: 0.5, similarity_boost: 0.75, speed: 1.0 }
    };

    const settings = styleSettings[voiceStyle] || styleSettings.documentary;

    try {
      fal.config({
        credentials: this.apiKey
      });

      const maxChunkLength = 4000;
      const chunks = [];
      let currentChunk = '';

      const sentences = text.split(/(?<=[.!?])\s+/);
      for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxChunkLength) {
          if (currentChunk) chunks.push(currentChunk.trim());
          currentChunk = sentence;
        } else {
          currentChunk += ' ' + sentence;
        }
      }
      if (currentChunk.trim()) chunks.push(currentChunk.trim());

      const audioUrls = [];

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];

        this.messageBus.setThinking('orchestrator', true, `Generating audio... (${i + 1}/${chunks.length})`);

        const result = await fal.subscribe("fal-ai/elevenlabs/tts/eleven-v3", {
          input: {
            text: chunk,
            voice: voiceId,
            stability: settings.stability,
            similarity_boost: settings.similarity_boost,
            speed: settings.speed
          },
          logs: true
        });

        if (result.data?.audio?.url) {
          audioUrls.push({
            url: result.data.audio.url,
            duration: result.data.audio.duration,
            contentType: result.data.audio.content_type
          });
        }
      }

      if (audioUrls.length > 0) {
        return {
          success: true,
          url: audioUrls[0].url,
          duration: audioUrls.reduce((sum, a) => sum + (a.duration || 0), 0),
          contentType: audioUrls[0].contentType,
          chunks: audioUrls
        };
      }

      return { success: false, error: 'Could not generate audio file' };

    } catch (error) {
      console.error('Voiceover generation error:', error);
      return { success: false, error: error.message };
    }
  }

  getState() {
    return {
      isRunning: this.isRunning,
      currentPhase: this.currentPhase,
      topic: this.topic,
      waitingForUser: this.waitingForUser,
      pendingQuestion: this.pendingQuestion,
      script: this.messageBus.getScript(),
      messages: this.messageBus.getHistory(),
      agents: Object.keys(this.agents).map(id => ({
        id,
        ...AGENT_PROFILES[id]
      }))
    };
  }

  reset() {
    this.messageBus.clear();
    this.isRunning = false;
    this.currentPhase = 'idle';
    this.topic = null;
    this.waitingForUser = false;
    this.pendingQuestion = null;
    this.userPreferences = {};
    this.finalScript = null;
    this.currentIteration = 0;

    // Reset duration tracking
    this.targetDuration = null;
    this.targetWordCount = { min: 0, max: 0 };

    for (const agent of Object.values(this.agents)) {
      agent.conversationHistory = [];
    }
  }
}

// ============================================
// SESSION MANAGEMENT
// ============================================

const sessions = new Map();

function getOrCreateSession(sessionId, apiKey) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, new OrchestrationEngine(apiKey));
  }
  return sessions.get(sessionId);
}

// ============================================
// SSE (Server-Sent Events) for real-time updates
// ============================================

const sseClients = new Map();

function sendSSE(sessionId, data) {
  const clients = sseClients.get(sessionId) || [];
  const message = `data: ${JSON.stringify(data)}\n\n`;
  clients.forEach(res => res.write(message));
}

// ============================================
// API ENDPOINTS
// ============================================

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/events/:sessionId', (req, res) => {
  const { sessionId } = req.params;

  // Validate session ID
  if (!SecurityUtils.isValidSessionId(sessionId)) {
    return res.status(400).json({ error: 'Invalid session ID format' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  if (!sseClients.has(sessionId)) {
    sseClients.set(sessionId, []);
  }
  sseClients.get(sessionId).push(res);

  req.on('close', () => {
    const clients = sseClients.get(sessionId) || [];
    const index = clients.indexOf(res);
    if (index !== -1) clients.splice(index, 1);
  });
});

// Check if server has API key configured
app.get('/api/config', (req, res) => {
  res.json({ hasServerApiKey: !!FAL_KEY });
});

app.post('/api/project/start', async (req, res) => {
  try {
    const { topic, context } = req.body;
    let { sessionId } = req.body;

    // Use API key from request body or fall back to server config
    const apiKey = req.body.apiKey || FAL_KEY;

    // Validate API key
    if (!apiKey) {
      return res.status(400).json({ error: 'API key required' });
    }
    if (!SecurityUtils.isValidApiKeyFormat(apiKey)) {
      return res.status(400).json({ error: 'Invalid API key format' });
    }

    // Validate topic
    if (!topic) {
      return res.status(400).json({ error: 'Topic required' });
    }

    // Sanitize inputs
    const sanitizedTopic = SecurityUtils.sanitizeInput(topic);
    const sanitizedContext = SecurityUtils.sanitizeInput(context || '');

    if (sanitizedTopic.length < 3) {
      return res.status(400).json({ error: 'Topic too short (min 3 characters)' });
    }

    // Generate or validate session ID
    if (!sessionId) {
      sessionId = SecurityUtils.generateSessionId();
    } else if (!SecurityUtils.isValidSessionId(sessionId)) {
      return res.status(400).json({ error: 'Invalid session ID format' });
    }

    // Rate limiting
    if (!SecurityUtils.checkRateLimit(sessionId)) {
      return res.status(429).json({ error: 'Rate limit exceeded. Please wait a moment.' });
    }

    const engine = getOrCreateSession(sessionId, apiKey);
    engine.reset();

    // Setup message listener for SSE
    engine.messageBus.on('message', (msg) => {
      sendSSE(sessionId, { type: 'message', data: msg });
    });

    engine.messageBus.on('script_update', (script) => {
      sendSSE(sessionId, { type: 'script', data: script });
    });

    engine.messageBus.on('thinking', (data) => {
      sendSSE(sessionId, { type: 'thinking', data });
    });

    engine.messageBus.on('phase', (data) => {
      sendSSE(sessionId, { type: 'phase', data });
    });

    await engine.start(sanitizedTopic, sanitizedContext);

    res.json({ ...engine.getState(), sessionId });
  } catch (error) {
    console.error('Start error:', error);
    res.status(500).json({ error: 'Failed to start project. Please try again.' });
  }
});

app.post('/api/project/respond', async (req, res) => {
  try {
    const { message, sessionId } = req.body;

    // Use API key from request body or fall back to server config
    const apiKey = req.body.apiKey || FAL_KEY;

    // Validate API key
    if (!apiKey) {
      return res.status(400).json({ error: 'API key required' });
    }

    // Validate session ID
    if (!sessionId || !SecurityUtils.isValidSessionId(sessionId)) {
      return res.status(400).json({ error: 'Invalid session ID' });
    }

    // Rate limiting
    if (!SecurityUtils.checkRateLimit(sessionId)) {
      return res.status(429).json({ error: 'Rate limit exceeded. Please wait a moment.' });
    }

    const engine = sessions.get(sessionId);
    if (!engine) {
      return res.status(404).json({ error: 'Session not found' });
    }

    // Sanitize message
    const sanitizedMessage = SecurityUtils.sanitizeInput(message || '');
    if (sanitizedMessage.length < 1) {
      return res.status(400).json({ error: 'Message cannot be empty' });
    }

    await engine.processUserResponse(sanitizedMessage);

    res.json(engine.getState());
  } catch (error) {
    console.error('Respond error:', error);
    res.status(500).json({ error: 'Failed to process response. Please try again.' });
  }
});

app.post('/api/project/continue', async (req, res) => {
  try {
    const { sessionId } = req.body;

    // Validate session ID
    if (!sessionId || !SecurityUtils.isValidSessionId(sessionId)) {
      return res.status(400).json({ error: 'Invalid session ID' });
    }

    // Rate limiting
    if (!SecurityUtils.checkRateLimit(sessionId)) {
      return res.status(429).json({ error: 'Rate limit exceeded. Please wait a moment.' });
    }

    const engine = sessions.get(sessionId);
    if (!engine) {
      return res.status(404).json({ error: 'Session not found' });
    }

    await engine.continueOrchestration();

    res.json(engine.getState());
  } catch (error) {
    console.error('Continue error:', error);
    res.status(500).json({ error: 'Failed to continue orchestration.' });
  }
});

app.get('/api/project/state/:sessionId', (req, res) => {
  const { sessionId } = req.params;

  // Validate session ID
  if (!SecurityUtils.isValidSessionId(sessionId)) {
    return res.status(400).json({ error: 'Invalid session ID format' });
  }

  const engine = sessions.get(sessionId);
  if (!engine) {
    return res.status(404).json({ error: 'Session not found' });
  }

  res.json(engine.getState());
});

app.post('/api/project/reset', (req, res) => {
  const { sessionId } = req.body;

  // Validate session ID
  if (!sessionId || !SecurityUtils.isValidSessionId(sessionId)) {
    return res.status(400).json({ error: 'Invalid session ID' });
  }

  const engine = sessions.get(sessionId);
  if (engine) {
    engine.reset();
  }

  res.json({ success: true });
});

app.get('/api/agents', (req, res) => {
  // Return only safe public agent info
  const safeAgentInfo = Object.entries(AGENT_PROFILES).reduce((acc, [id, profile]) => {
    acc[id] = {
      id: profile.id,
      name: profile.name,
      emoji: profile.emoji,
      color: profile.color,
      role: profile.role
    };
    return acc;
  }, {});
  res.json(safeAgentInfo);
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'An unexpected error occurred' });
});

// ============================================
// START SERVER
// ============================================

app.listen(PORT, () => {
  console.log(`
  ╔═══════════════════════════════════════════════════════╗
  ║     YouTube Script Generator - Multi-Agent System     ║
  ║     Running on http://localhost:${PORT}                  ║
  ╚═══════════════════════════════════════════════════════╝

  Model: ${AGENT_MODEL}
  Search Model: ${SEARCH_MODEL}

  Agents:
  🎬 Producer      - Executive Producer
  🔍 Researcher    - Content Researcher
  ✍️  Scriptwriter  - YouTube Scriptwriter
  🎭 Editor        - Content Editor
  ✅ Fact-Checker  - Fact Verification
  💡 Creative      - Creative Director
  🎙️ Voice Artist  - Voiceover Producer

  Endpoints:
  - POST /api/project/start   - Start new project
  - POST /api/project/respond - Send user response
  - POST /api/project/continue - Continue orchestration
  - GET  /api/project/state   - Get current state
  - GET  /api/events/:id      - SSE for real-time updates
  `);
});

export default app;
