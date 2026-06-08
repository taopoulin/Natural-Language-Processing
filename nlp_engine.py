"""Tao.ai own NLP engine — no external LLM required."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NLP = None
_NLP_LOAD_ERROR: str | None = None

POSITIVE_WORDS = {
    "good", "great", "awesome", "happy", "love", "excellent", "nice",
    "wonderful", "amazing", "fantastic", "best", "thanks", "thank",
    "glad", "enjoy", "beautiful", "perfect", "fun", "cool",
}
NEGATIVE_WORDS = {
    "bad", "sad", "angry", "hate", "terrible", "awful", "upset",
    "worst", "horrible", "annoyed", "frustrated", "disappointed",
    "boring", "slow", "broken", "wrong", "fail", "failed",
}
STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "that", "this", "it", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "be", "are", "was", "were", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "about", "from", "as", "at", "by",
    "not", "but", "if", "so", "what", "how", "when", "where", "who",
    "why", "which", "me", "him", "her", "them", "us",
}
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|yo)\b",
    re.I,
)
QUESTION_STARTERS = re.compile(
    r"^(what|who|where|when|why|how|is|are|can|could|would|do|does|did)\b",
    re.I,
)


def _load_spacy():
    global _NLP, _NLP_LOAD_ERROR
    if _NLP is not None or _NLP_LOAD_ERROR is not None:
        return _NLP
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    except Exception as exc:
        _NLP_LOAD_ERROR = str(exc)
        _NLP = None
    return _NLP


def _tokenize(text: str) -> list[str]:
    nlp = _load_spacy()
    if nlp:
        return [t.text for t in nlp(text) if not t.is_space]
    return text.split()


def _clean_word(word: str) -> str:
    return word.strip(".,!?;:\"'()[]{}").lower()


@dataclass
class NLPResult:
    text: str
    sentiment: str
    sentiment_score: float
    keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_count: int = 0
    char_count: int = 0
    language: str = "en"
    is_question: bool = False
    is_greeting: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "sentiment_score": round(self.sentiment_score, 3),
            "keywords": self.keywords,
            "entities": self.entities,
            "tokens": self.tokens[:50],
            "token_count": self.token_count,
            "char_count": self.char_count,
            "language": self.language,
            "is_question": self.is_question,
            "is_greeting": self.is_greeting,
        }


def analyze_sentiment(text: str) -> tuple[str, float]:
    words = {_clean_word(w) for w in text.split()}
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return "neutral", 0.0
    score = (pos - neg) / total
    if score > 0.15:
        return "positive", score
    if score < -0.15:
        return "negative", score
    return "neutral", score


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    nlp = _load_spacy()
    if nlp:
        doc = nlp(text)
        keywords = [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha
            and not t.is_stop
            and len(t.text) > 2
            and t.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
        ]
        return list(dict.fromkeys(keywords))[:limit]

    words = [_clean_word(w) for w in text.split()]
    keywords = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
    return list(dict.fromkeys(keywords))[:limit]


def extract_entities(text: str) -> list[dict[str, str]]:
    nlp = _load_spacy()
    if nlp:
        doc = nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    entities: list[dict[str, str]] = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        entities.append({"text": match.group(), "label": "PROPER_NOUN", "start": match.start(), "end": match.end()})
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
        entities.append({"text": match.group(), "label": "DATE", "start": match.start(), "end": match.end()})
    for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text):
        entities.append({"text": match.group(), "label": "NUMBER", "start": match.start(), "end": match.end()})
    return entities


def analyze(text: str) -> NLPResult:
    text = text.strip()
    sentiment, score = analyze_sentiment(text)
    tokens = _tokenize(text)
    return NLPResult(
        text=text,
        sentiment=sentiment,
        sentiment_score=score,
        keywords=extract_keywords(text),
        entities=extract_entities(text),
        tokens=tokens,
        token_count=len(tokens),
        char_count=len(text),
        is_question=text.rstrip().endswith("?") or bool(QUESTION_STARTERS.match(text)),
        is_greeting=bool(GREETING_PATTERNS.match(text)),
    )


def summarize(text: str, max_sentences: int = 2) -> str:
    nlp = _load_spacy()
    if nlp:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        scored: list[tuple[float, str]] = []
        for sent in doc.sents:
            score = sum(1 for t in sent if t.pos_ in {"NOUN", "PROPN", "VERB"} and not t.is_stop)
            scored.append((score, sent.text.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scored[:max_sentences], key=lambda x: text.find(x[1]))
        return " ".join(s for _, s in top)

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sentences]) if parts else text


FOLLOW_UP_PATTERNS = re.compile(
    r"\b(that|this|it|those|these|same|more|again|earlier|before|you said|what about)\b",
    re.I,
)


@dataclass
class ConversationContext:
    user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    turn_count: int = 0
    all_keywords: list[str] = field(default_factory=list)
    all_entities: list[dict[str, str]] = field(default_factory=list)
    sentiments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "all_keywords": self.all_keywords,
            "all_entities": self.all_entities,
            "sentiments": self.sentiments,
        }


def build_conversation_context(messages: list[dict[str, str]]) -> ConversationContext:
    ctx = ConversationContext()
    seen_keywords: set[str] = set()
    seen_entities: set[str] = set()

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            ctx.user_messages.append(content)
            ctx.turn_count += 1
            result = analyze(content)
            ctx.sentiments.append(result.sentiment)
            for kw in result.keywords:
                if kw not in seen_keywords:
                    seen_keywords.add(kw)
                    ctx.all_keywords.append(kw)
            for ent in result.entities:
                key = f"{ent['text']}:{ent['label']}"
                if key not in seen_entities:
                    seen_entities.add(key)
                    ctx.all_entities.append(ent)
        elif role == "assistant":
            ctx.assistant_messages.append(content)

    return ctx


def generate_reply(messages: list[dict[str, str]]) -> str:
    ctx = build_conversation_context(messages)
    last_user = ctx.user_messages[-1] if ctx.user_messages else ""

    if not last_user:
        return "I'm Tao.ai running on your own NLP API. Send a message and I'll analyze and respond."

    result = analyze(last_user)
    is_follow_up = bool(FOLLOW_UP_PATTERNS.search(last_user)) and ctx.turn_count > 1
    lines: list[str] = []

    if result.is_greeting and ctx.turn_count == 1:
        lines.append("Hello! I'm Tao.ai, powered by your own NLP API (no external LLM required).")
    elif ctx.turn_count > 1:
        lines.append(f"Continuing our conversation (message {ctx.turn_count} from you).")

    if is_follow_up and ctx.all_keywords:
        prior_topics = ", ".join(ctx.all_keywords[:5])
        lines.append(f"Picking up from earlier topics: {prior_topics}.")

    if ctx.turn_count > 1 and len(ctx.user_messages) >= 2:
        prev_user = ctx.user_messages[-2]
        if len(prev_user) > 0:
            lines.append(f'You previously said: "{prev_user[:120]}{"..." if len(prev_user) > 120 else ""}"')

    if result.is_question:
        lines.append("I detected a question in your message.")

    sentiment_note = {
        "positive": "Your message sounds positive.",
        "negative": "Your message sounds negative — I'm here to help.",
        "neutral": "Your message has a neutral tone.",
    }[result.sentiment]
    lines.append(sentiment_note)

    if len(ctx.sentiments) > 1:
        trend = " → ".join(ctx.sentiments[-4:])
        lines.append(f"Sentiment across this chat: {trend}.")

    if result.keywords:
        lines.append(f"Key topics in this message: {', '.join(result.keywords)}.")
    elif is_follow_up and ctx.all_keywords:
        lines.append(f"Related topics from our history: {', '.join(ctx.all_keywords[:6])}.")

    if result.entities:
        ent_parts = [f"{e['text']} ({e['label']})" for e in result.entities[:5]]
        lines.append(f"Entities found: {', '.join(ent_parts)}.")
    elif ctx.all_entities:
        ent_parts = [f"{e['text']} ({e['label']})" for e in ctx.all_entities[:5]]
        lines.append(f"Entities from our conversation: {', '.join(ent_parts)}.")

    if len(last_user.split()) > 30:
        summary = summarize(last_user)
        lines.append(f"Summary: {summary}")
    elif ctx.turn_count > 2:
        history_text = " ".join(ctx.user_messages)
        if len(history_text.split()) > 40:
            lines.append(f"Conversation summary: {summarize(history_text, max_sentences=3)}")

    if result.is_question and (result.keywords or ctx.all_keywords):
        topic = result.keywords[0] if result.keywords else ctx.all_keywords[0]
        lines.append(
            f"Regarding **{topic}**: I used your full chat history plus local NLP "
            f"(sentiment, keywords, entities, summarization) to shape this reply."
        )
    elif not result.is_greeting or ctx.turn_count > 1:
        lines.append(
            f"This reply used **{ctx.turn_count}** user message(s) from history — "
            "all processed by your own Tao.ai NLP API."
        )

    return "\n\n".join(lines)


def engine_info() -> dict[str, Any]:
    nlp = _load_spacy()
    return {
   
    /* change the name here */
        "name": "Tao.ai NLP",
        "version": "1.0.0",
        "spacy_available": nlp is not None,
        "spacy_model": "en_core_web_sm" if nlp else None,
        "spacy_error": _NLP_LOAD_ERROR,
        "capabilities": [
            "sentiment",
            "keywords",
            "entities",
            "tokenize",
            "summarize",
            "chat",
        ],
    }
