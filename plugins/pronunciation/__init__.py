"""Pronunciation plugin — IPA transcription + connected speech + TTS playback."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_panel_ref = [None]


# -- HTML template -----------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
:root {{
  --bg: #f5f5f7; --text: #1d1d1f; --secondary: #86868b;
  --card: #ffffff; --border: #d2d2d7; --accent: #007aff;
  --link: #168baf; --link-soft: rgba(22, 139, 175, 0.09);
  --playback-height: 140px;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1c1c1e; --text: #e5e5e7; --secondary: #98989d;
    --card: #2c2c2e; --border: #48484a; --accent: #0a84ff;
    --link: #52cbd6; --link-soft: rgba(82, 203, 214, 0.10);
  }}
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; overflow: hidden; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  user-select: text; -webkit-user-select: text;
}}
/* Pin both regions to the viewport instead of relying on document height. */
.content-scroll {{
  position: fixed; inset: 0 0 var(--playback-height) 0;
  overflow-y: auto; padding: 32px 40px 24px;
}}
.container {{ max-width: 720px; margin: 0 auto; }}

.sentence {{
  font-size: 26px; font-weight: 600; text-align: center;
  margin-bottom: 28px; line-height: 1.5;
}}
.rhythm-controls {{ text-align:center; margin:-12px 0 24px; }}
.rhythm-controls .audio-btn {{ font-size:13px; padding:8px 14px; }}
.rhythm-controls .rhythm-summary {{ max-width:560px; margin:6px auto; }}
.rhythm-status {{ font-size:12px; margin-top:8px; color:var(--secondary); }}

.section {{ margin-bottom: 24px; }}
.label {{
  font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--secondary); margin-bottom: 14px; text-align: center;
}}

.words {{
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 6px 28px;
}}
.words-stage {{ position: relative; }}
.words-stage.has-links .words {{ row-gap: 16px; }}
.words-stage.has-links .word-item {{ padding-bottom: 25px; }}
.word-item {{ text-align: center; }}
.word-text {{ font-size: 19px; font-weight: 500; line-height: 1.4; }}
.word-ipa {{
  font-size: 15px; color: var(--secondary); margin-top: 2px; line-height: 1.3;
  transition: color 0.2s, text-shadow 0.2s;
}}
.word-item.link-active .word-ipa {{
  color: var(--link); text-shadow: 0 0 14px var(--link-soft);
}}
.link-ropes {{
  position: absolute; inset: 0; width: 100%; height: 100%;
  overflow: visible; pointer-events: none;
}}
.link-rope {{
  fill: none; stroke: var(--link); stroke-width: 1.6;
  stroke-linecap: round; opacity: 0.6;
  transition: opacity 0.2s, stroke-width 0.2s, filter 0.2s;
}}
.link-rope.draw-in {{ animation: rope-draw 0.65s ease backwards; }}
.link-flow {{
  fill: none; stroke: var(--accent); stroke-width: 3.5; stroke-linecap: round;
  stroke-dasharray: 12 88; animation: rope-flow 2.6s linear infinite;
  filter: drop-shadow(0 0 3px var(--link)); pointer-events: none;
}}
.link-dot {{ fill: var(--link); animation: rope-breathe 2.6s ease-in-out infinite; }}
.page-hidden .link-flow, .page-hidden .link-dot {{ animation-play-state: paused; }}
.link-replay {{
  color: var(--link); background: var(--link-soft); border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 10px; margin: 0 0 10px; cursor: pointer;
  font: inherit; font-size: 11px;
}}
@keyframes rope-flow {{ from {{ stroke-dashoffset: 100; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes rope-breathe {{ 0%, 100% {{ opacity: 0.45; }} 50% {{ opacity: 1; }} }}
.link-hit {{
  fill: none; stroke: transparent; stroke-width: 16;
  pointer-events: stroke; cursor: pointer;
}}
.link-group.link-active .link-rope {{
  opacity: 1; stroke-width: 2.2;
  filter: drop-shadow(0 0 3px var(--link-soft));
}}
.link-marker {{ fill: var(--link); font-size: 9px; text-anchor: middle; }}
.links-panel {{ text-align: center; margin-top: 4px; }}
.links-panel .label {{ margin-bottom: 10px; color: var(--link); }}
.link-choices {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; }}
.link-choice {{
  font: inherit; font-size: 12px; color: var(--secondary);
  background: transparent; border: 1px solid var(--border);
  border-radius: 20px; padding: 5px 10px; cursor: pointer;
  transition: color 0.2s, background 0.2s, border-color 0.2s;
}}
.link-choice.link-active {{
  color: var(--link); background: var(--link-soft); border-color: var(--link);
}}
button:focus-visible {{ outline: 2px solid var(--link); outline-offset: 3px; }}
.link-note {{
  min-height: 18px; max-width: 540px; margin: 8px auto 0;
  font-size: 12px; line-height: 1.5; color: var(--text);
}}
.link-caption {{ color: var(--secondary); font-size: 11px; line-height: 1.5; }}
.speech-panel {{ margin-top: 20px; text-align: center; }}
.speech-panel .label {{ margin-bottom: 10px; }}
.speech-detail {{ font-size: 13px; line-height: 1.6; margin-top: 12px; }}
.speech-ipa {{ color: var(--accent); font-size: 18px; margin-bottom: 5px; }}
.speech-active .word-ipa {{ color: var(--accent); text-shadow: 0 0 14px var(--link-soft); }}
.speech-choice[aria-pressed="true"] {{ color: var(--accent); border-color: var(--accent); background: var(--link-soft); }}
.reveal {{ animation: content-reveal 0.35s ease both; }}
@keyframes rope-draw {{ from {{ stroke-dashoffset: 1; opacity: 0; }} to {{ stroke-dashoffset: 0; opacity: 0.6; }} }}
@keyframes content-reveal {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}

.connected {{
  font-size: 21px; text-align: center; font-weight: 500;
  color: var(--accent);
  padding: 16px 28px;
  background: var(--card); border-radius: 12px;
  border: 1px solid var(--border);
  line-height: 1.6;
}}

.translation {{
  font-size: 15px; text-align: center; color: var(--secondary);
  margin-top: 10px; line-height: 1.6;
}}

.playback-bar {{
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 10;
  padding: 14px 24px;
  background: var(--bg); border-top: 1px solid var(--border);
}}
.playback-bar .error {{ overflow-wrap: anywhere; }}
.audio-row {{
  display: flex; flex-wrap: wrap; justify-content: center; gap: 12px 24px;
}}
.voice-controls {{ display: flex; align-items: center; gap: 8px; }}
.audio-btn {{
  display: inline-flex; align-items: center; gap: 8px;
  padding: 10px 28px; border-radius: 10px;
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-size: 15px; cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
}}
.audio-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
.audio-btn:disabled {{ opacity: 0.35; cursor: default; }}
.audio-btn:disabled:hover {{ border-color: var(--border); color: var(--text); }}
.audio-btn .icon {{ font-size: 18px; }}
.save-btn {{ padding: 10px 14px; min-width: 0; }}

.loading {{
  text-align: center; color: var(--secondary); font-size: 14px;
}}
.spinner {{
  width: 20px; height: 20px;
  border: 2.5px solid var(--border); border-top-color: var(--accent);
  border-radius: 50%; animation: spin 0.7s linear infinite;
  margin: 0 auto 10px;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

.error {{ color: #ff453a; text-align: center; font-size: 13px; margin-top: 8px; }}
.hidden {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{
  :root:not(.motion-replay) *, :root:not(.motion-replay) *::before, :root:not(.motion-replay) *::after {{
    animation: none !important; transition: none !important;
  }}
  :root:not(.motion-replay) .link-flow {{ display: none; }}
  .motion-replay .link-flow, .motion-replay .link-dot {{ animation-iteration-count: 1; }}
}}
</style>
</head>
<body>
<main class="content-scroll" tabindex="0" aria-label="Pronunciation content">
<div class="container">

  <div class="sentence" id="sentence">{sentence}</div>
  <div class="rhythm-controls">
    <button class="audio-btn" id="rhythm-action" type="button">Mark stress &amp; pauses</button>
    <div class="rhythm-status" id="rhythm-status" role="status" aria-live="polite"></div>
    <div class="rhythm-legend hidden" id="rhythm-legend">
      <span class="rhythm-stress">Bold color = stress</span> · | brief pause · || longer pause<br>
      Suggested phrasing, not audio timing. Stress does not require a pause.
    </div>
    <div class="rhythm-summary hidden" id="rhythm-summary"></div>
  </div>

  <div id="ph-content" class="hidden">
    <div class="section">
      <div class="label">Pronunciation</div>
      <div class="words-stage" id="words-stage">
        <div class="words" id="words"></div>
        <svg class="link-ropes" id="link-ropes" aria-hidden="true"></svg>
      </div>
      <div class="links-panel hidden" id="links-panel">
        <div class="label">Suggested links</div>
        <button class="link-replay" id="replay-links" type="button">Replay links</button>
        <div class="link-choices" id="link-choices"></div>
        <div class="link-note" id="link-note"></div>
        <div class="link-caption">Suggested readings, not audio timing.</div>
      </div>
    </div>
  </div>

  <div id="ph-error" class="error hidden"></div>

  <div class="section" id="connected-section">
    <div class="label">Connected Speech</div>
    <div style="text-align: center;" id="connected-btn-wrap">
      <button class="audio-btn" id="btn-connected" onclick="requestConnected()">
        Analyze with AI
      </button>
    </div>
    <div id="connected-loading" class="loading hidden">
      <div class="spinner"></div>Analyzing…
    </div>
    <div class="connected hidden" id="connected"></div>
    <div class="translation hidden" id="translation"></div>
    <div class="speech-panel hidden" id="speech-panel">
      <div class="label">American speech</div>
      <div class="link-choices" id="speech-choices"></div>
      <div class="speech-detail" id="speech-detail" aria-live="polite">
        <div class="speech-ipa" id="speech-ipa"></div>
        <div id="speech-note"></div>
      </div>
    </div>
    <div id="connected-error" class="error hidden" role="alert"></div>
  </div>

</div>
</main>

<footer class="playback-bar" aria-label="Audio playback">
  <div class="container">
    <div class="audio-row">
      <div class="voice-controls" role="group" aria-label="Female voice">
        <button class="audio-btn" id="btn-f" disabled onclick="playF()">
          <span class="icon">&#9792;</span> Female
        </button>
        <button class="audio-btn save-btn" id="save-f" disabled onclick="saveF()" title="Save">&#8595;</button>
      </div>
      <div class="voice-controls" role="group" aria-label="Male voice">
        <button class="audio-btn" id="btn-m" disabled onclick="playM()">
          <span class="icon">&#9794;</span> Male
        </button>
        <button class="audio-btn save-btn" id="save-m" disabled onclick="saveM()" title="Save">&#8595;</button>
      </div>
    </div>
    <div id="tts-loading" class="loading" style="margin-top: 10px;">
      <div class="spinner"></div>Generating audio…
    </div>
    <div id="tts-error" class="error hidden"></div>
  </div>
</footer>
<script>
var _playbackBar = document.querySelector(".playback-bar");
var _playbackHeight = null;
function _syncPlaybackHeight() {{
  var height = _playbackBar.offsetHeight;
  if (height === _playbackHeight) return;
  _playbackHeight = height;
  // Keep the last content line reachable when buttons wrap or status text changes.
  document.documentElement.style.setProperty("--playback-height", height + "px");
}}
_syncPlaybackHeight();
new ResizeObserver(_syncPlaybackHeight).observe(_playbackBar);

var _audio = {{}};
var _words = [];
var _links = [];
var _selectedLink = null;
var _linkFrame = null;
var _animateLinks = false;
var _linkLayout = "";
var _analyzing = false;
var _features = [];
var _rhythm = null;
var _rhythmRequest = 0;
var _allLinks = [];

function _rhythmWordsAligned() {{
  function normalized(word) {{ return word.toLowerCase().replace(/[^a-z0-9]/g, ""); }}
  return _rhythm && _words.length === _rhythm.words.length
    && _words.every(function(word, i) {{ return normalized(word.word) === normalized(_rhythm.words[i]); }});
}}
function _applyWordStress() {{
  var items = document.getElementById("words").children;
  var marked = new Map((_rhythm ? _rhythm.stress : []).map(function(item) {{ return [item.index, item]; }}));
  var aligned = _rhythmWordsAligned();
  Array.from(items).forEach(function(item, index) {{
    var word = item.querySelector(".word-text");
    var hint = aligned && marked.get(index);
    word.classList.toggle("rhythm-stress", !!hint);
    word.title = hint ? hint.note : "";
  }});
}}

document.getElementById("rhythm-action").addEventListener("click", function() {{
  var button = document.getElementById("rhythm-action");
  if (button.disabled) return;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Marking…";
  document.getElementById("rhythm-status").className = "rhythm-status";
  document.getElementById("rhythm-status").textContent = "Finding emphasis and thought groups…";
  wz.send("request_rhythm", {{request_id: ++_rhythmRequest}});
}});
wz.on("rhythm_progress", function(data) {{
  if (data.request_id !== _rhythmRequest) return;
  document.getElementById("rhythm-status").textContent = "Marking passage " + data.done + " / " + data.total + "…";
}});
wz.on("rhythm_result", function(data) {{
  if (data.request_id !== _rhythmRequest) return;
  _rhythm = data.result;
  PhRhythm.render(document.getElementById("sentence"), __SENTENCE_JS__, _rhythm);
  _applyWordStress();
  _renderLinks(_allLinks);
  document.getElementById("rhythm-legend").classList.remove("hidden");
  document.getElementById("rhythm-summary").classList.toggle("hidden", !_rhythm.summary);
  document.getElementById("rhythm-summary").textContent = _rhythm.summary;
  document.getElementById("rhythm-status").textContent = "";
  var button = document.getElementById("rhythm-action");
  button.disabled = false;
  button.setAttribute("aria-busy", "false");
  button.textContent = "Mark stress & pauses again";
}});
wz.on("rhythm_error", function(data) {{
  if (data.request_id !== _rhythmRequest) return;
  var status = document.getElementById("rhythm-status");
  status.className = "rhythm-error";
  status.textContent = data.message;
  var button = document.getElementById("rhythm-action");
  button.disabled = false;
  button.setAttribute("aria-busy", "false");
  button.textContent = "Retry stress & pauses";
}});

function _renderWords(words) {{
  _words = words;
  var w = document.getElementById("words");
  w.innerHTML = "";
  words.forEach(function(item) {{
    var el = document.createElement("div");
    el.className = "word-item";
    el.innerHTML = '<div class="word-text">' + _e(item.word)
      + '</div><div class="word-ipa">' + _e(item.ipa) + '</div>';
    w.appendChild(el);
  }});
  _applyWordStress();
}}

function _highlightLink(index) {{
  document.querySelectorAll(".link-active").forEach(function(el) {{
    el.classList.remove("link-active");
  }});
  document.querySelectorAll("#link-choices .link-choice").forEach(function(el) {{
    el.setAttribute("aria-pressed", String(Number(el.dataset.linkIndex) === _selectedLink));
  }});
  var link = _links[index];
  document.getElementById("link-note").textContent = link ? link.note : "Hover over a curve or select a pair.";
  if (!link) return;
  var items = document.getElementById("words").children;
  items[link.from].classList.add("link-active");
  items[link.to].classList.add("link-active");
  document.querySelectorAll('[data-link-index="' + index + '"]').forEach(function(el) {{
    el.classList.add("link-active");
  }});
}}

function _svgElement(name, attributes) {{
  var el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.keys(attributes).forEach(function(key) {{ el.setAttribute(key, attributes[key]); }});
  return el;
}}

function _drawLinks(animate) {{
  var svg = document.getElementById("link-ropes");
  var stage = document.getElementById("words-stage").getBoundingClientRect();
  var items = document.getElementById("words").children;
  var bounds = _links.map(function(link) {{
    return [items[link.from].querySelector(".word-ipa").getBoundingClientRect(),
      items[link.to].querySelector(".word-ipa").getBoundingClientRect()];
  }});
  var layout = JSON.stringify([stage.width, stage.height, bounds]);
  // The first resize notification must not replace ropes that are still drawing.
  if (!animate && layout === _linkLayout) return;
  _linkLayout = layout;
  svg.replaceChildren();
  _links.forEach(function(link, index) {{
    var a = bounds[index][0];
    var b = bounds[index][1];
    var x1 = a.right - stage.left - Math.min(10, a.width / 4);
    var x2 = b.left - stage.left + Math.min(10, b.width / 4);
    var y1 = a.bottom - stage.top + 4;
    var y2 = b.bottom - stage.top + 4;
    var group = _svgElement("g", {{"class": "link-group", "data-link-index": index}});
    var path;
    if (Math.abs(a.top - b.top) < 4 && x2 > x1) {{
      var depth = Math.min(22, Math.max(12, (x2 - x1) * 0.22));
      path = "M " + x1 + " " + y1 + " C " + x1 + " " + (y1 + depth)
        + " " + x2 + " " + (y2 + depth) + " " + x2 + " " + y2;
    }} else {{
      // A wrapped pair gets matching numbered ends instead of a cross-row rope.
      path = "M " + (x1 - 9) + " " + y1 + " q 0 9 9 9"
        + " M " + (x2 + 9) + " " + y2 + " q 0 9 -9 9";
      [[x1 + 6, y1 + 12], [x2 - 6, y2 + 12]].forEach(function(point) {{
        var marker = _svgElement("text", {{"x": point[0], "y": point[1], "class": "link-marker"}});
        marker.textContent = index + 1;
        group.appendChild(marker);
      }});
    }}
    var rope = _svgElement("path", {{
      "d": path, "pathLength": "1", "class": "link-rope" + (animate ? " draw-in" : "")
    }});
    if (animate) {{
      rope.style.strokeDasharray = "1";
      rope.style.animationDelay = (index * 110) + "ms";
    }}
    group.appendChild(rope);
    var flow = _svgElement("path", {{"d": path, "pathLength": "100", "class": "link-flow"}});
    flow.style.animationDelay = (index * 220) + "ms";
    group.appendChild(flow);
    [[x1, y1], [x2, y2]].forEach(function(point) {{
      group.appendChild(_svgElement("circle", {{"cx": point[0], "cy": point[1], "r": "2.5", "class": "link-dot"}}));
    }});
    group.appendChild(_svgElement("path", {{"d": path, "class": "link-hit"}}));
    svg.appendChild(group);
  }});
  _highlightLink(_selectedLink);
}}

function _scheduleLinks(animate) {{
  _animateLinks = _animateLinks || animate;
  if (_linkFrame !== null) cancelAnimationFrame(_linkFrame);
  _linkFrame = requestAnimationFrame(function() {{
    _linkFrame = null;
    _drawLinks(_animateLinks);
    _animateLinks = false;
  }});
}}

function _renderLinks(links) {{
  _allLinks = Array.isArray(links) ? links : [];
  var pauses = new Set((_rhythmWordsAligned() ? _rhythm.pauses : []).map(function(item) {{ return item.after; }}));
  var seen = {{}};
  _links = _allLinks.filter(function(link) {{
    if (!link || !Number.isInteger(link.from) || link.to !== link.from + 1
        || link.from < 0 || link.to >= _words.length || seen[link.from]
        || typeof link.note !== "string" || !link.note.trim() || pauses.has(link.from)) return false;
    seen[link.from] = true;
    return true;
  }});
  _selectedLink = null;
  document.getElementById("words-stage").classList.toggle("has-links", _links.length > 0);
  document.getElementById("links-panel").classList.toggle("hidden", !_links.length);
  var choices = document.getElementById("link-choices");
  choices.replaceChildren();
  _links.forEach(function(link, index) {{
    var button = document.createElement("button");
    button.className = "link-choice";
    button.dataset.linkIndex = index;
    button.setAttribute("aria-pressed", "false");
    button.textContent = (index + 1) + " · " + _words[link.from].word + " ↝ " + _words[link.to].word;
    button.setAttribute("aria-label", button.textContent + ". " + link.note);
    choices.appendChild(button);
  }});
  _scheduleLinks(true);
}}

["words-stage", "link-choices"].forEach(function(id) {{
  var container = document.getElementById(id);
  function target(event) {{ return event.target.closest("[data-link-index]"); }}
  container.addEventListener("mouseover", function(event) {{
    var el = target(event);
    _highlightLink(el ? Number(el.dataset.linkIndex) : _selectedLink);
  }});
  container.addEventListener("mouseleave", function() {{ _highlightLink(_selectedLink); }});
  container.addEventListener("focusin", function(event) {{
    var el = target(event);
    if (el) _highlightLink(Number(el.dataset.linkIndex));
  }});
  container.addEventListener("focusout", function() {{ _highlightLink(_selectedLink); }});
  container.addEventListener("click", function(event) {{
    var el = target(event);
    if (!el) return;
    _selectedLink = Number(el.dataset.linkIndex);
    _highlightLink(_selectedLink);
  }});
}});
new ResizeObserver(function() {{ _scheduleLinks(false); }}).observe(document.getElementById("words"));
document.fonts.ready.then(function() {{ _scheduleLinks(false); }});
var _replayTimer = null;
document.getElementById("replay-links").addEventListener("click", function() {{
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {{
    // A deliberate replay allows one brief demonstration without changing the OS preference.
    clearTimeout(_replayTimer);
    document.documentElement.classList.add("motion-replay");
    _replayTimer = setTimeout(function() {{
      document.documentElement.classList.remove("motion-replay");
    }}, 2700 + _links.length * 220);
  }}
  _scheduleLinks(true);
}});
if (matchMedia("(prefers-reduced-motion: reduce)").matches) {{
  document.querySelector(".link-caption").textContent = "Reduced motion is on. Replay links animates once; timing is illustrative.";
}}
function _syncMotionVisibility() {{
  document.documentElement.classList.toggle("page-hidden", document.hidden);
}}
document.addEventListener("visibilitychange", _syncMotionVisibility);
_syncMotionVisibility();

function _selectFeature(index) {{
  var feature = _features[index];
  if (!feature) return;
  document.querySelectorAll(".word-item.speech-active").forEach(function(el) {{ el.classList.remove("speech-active"); }});
  var items = document.getElementById("words").children;
  feature.words.forEach(function(i) {{ items[i].classList.add("speech-active"); }});
  document.querySelectorAll(".speech-choice").forEach(function(button) {{
    button.setAttribute("aria-pressed", String(Number(button.dataset.featureIndex) === index));
  }});
  document.getElementById("speech-ipa").textContent = feature.ipa;
  document.getElementById("speech-note").textContent = feature.note;
}}

function _renderFeatures(features) {{
  var names = {{weak_form: "Weak form", flap: "Flap", unreleased_stop: "Unreleased stop"}};
  _features = (Array.isArray(features) ? features : []).filter(function(feature) {{
    return feature && Object.prototype.hasOwnProperty.call(names, feature.kind)
      && Array.isArray(feature.words) && feature.words.length > 0
      && feature.words.every(function(i) {{ return Number.isInteger(i) && i >= 0 && i < _words.length; }})
      && typeof feature.ipa === "string" && typeof feature.note === "string";
  }});
  var choices = document.getElementById("speech-choices");
  choices.replaceChildren();
  document.getElementById("speech-panel").classList.toggle("hidden", !_features.length);
  _features.forEach(function(feature, index) {{
    var button = document.createElement("button");
    button.className = "link-choice speech-choice";
    button.dataset.featureIndex = index;
    button.textContent = names[feature.kind] + " · " + feature.words.map(function(i) {{ return _words[i].word; }}).join(" ");
    button.setAttribute("aria-pressed", "false");
    choices.appendChild(button);
  }});
  if (_features.length) _selectFeature(0);
}}
document.getElementById("speech-choices").addEventListener("click", function(event) {{
  var button = event.target.closest("[data-feature-index]");
  if (button) _selectFeature(Number(button.dataset.featureIndex));
}});

function _e(s) {{
  var d = document.createElement("span");
  d.textContent = s;
  return d.innerHTML;
}}
function _show(id, msg) {{
  var el = document.getElementById(id);
  el.textContent = msg;
  el.classList.remove("hidden");
}}

wz.on("audio", function(d) {{
  document.getElementById("tts-loading").classList.add("hidden");
  _audio = d;
  document.getElementById("btn-f").disabled = false;
  document.getElementById("btn-m").disabled = false;
  document.getElementById("save-f").disabled = false;
  document.getElementById("save-m").disabled = false;
}});

wz.on("audio_error", function(d) {{
  document.getElementById("tts-loading").classList.add("hidden");
  _show("tts-error", d.message);
}});

function playF() {{ if (_audio.female) wz.playAudio(_audio.female); }}
function playM() {{ if (_audio.male) wz.playAudio(_audio.male); }}

function _saveAudio(dataUrl, suffix) {{
  var name = __SENTENCE_JS__.replace(/[^a-zA-Z0-9 ]/g, '').trim().substring(0, 40).trim();
  if (!name) name = 'audio';
  wz.send("save_audio", {{data: dataUrl, filename: name + '_' + suffix + '.mp3'}});
}}
function saveF() {{ if (_audio.female) _saveAudio(_audio.female, 'female'); }}
function saveM() {{ if (_audio.male) _saveAudio(_audio.male, 'male'); }}

wz.on("save_ok", function(d) {{
  _show("tts-error", "Saved to " + d.path);
  var el = document.getElementById("tts-error");
  el.style.color = "var(--secondary)";
  setTimeout(function() {{ el.classList.add("hidden"); el.style.color = ""; }}, 3000);
}});
wz.on("save_error", function(d) {{ _show("tts-error", d.message); }});

function requestConnected() {{
  if (_analyzing) return;
  _analyzing = true;
  document.getElementById("connected-btn-wrap").classList.add("hidden");
  var error = document.getElementById("connected-error");
  error.textContent = "";
  error.classList.add("hidden");
  var loading = document.getElementById("connected-loading");
  loading.classList.remove("hidden");
  loading.classList.add("reveal");
  wz.send("request_connected", {{}});
}}

wz.on("connected_speech", function(d) {{
  _analyzing = false;
  document.getElementById("connected-loading").classList.add("hidden");
  document.getElementById("connected-btn-wrap").classList.add("hidden");
  document.getElementById("connected-error").classList.add("hidden");
  if (d.words) {{
    _renderWords(d.words);
    document.getElementById("ph-content").classList.remove("hidden");
    document.getElementById("ph-error").classList.add("hidden");
  }}
  _renderLinks(d.links);
  _renderFeatures(d.features);
  var el = document.getElementById("connected");
  el.textContent = d.connected;
  el.classList.remove("hidden");
  el.classList.add("reveal");
  var tr = document.getElementById("translation");
  tr.classList.toggle("hidden", !d.translation);
  if (d.translation) {{
    tr.textContent = d.translation;
    tr.classList.add("reveal");
  }}
}});

wz.on("connected_error", function(d) {{
  _analyzing = false;
  document.getElementById("connected-loading").classList.add("hidden");
  document.getElementById("btn-connected").textContent = "Try again";
  document.getElementById("connected-btn-wrap").classList.remove("hidden");
  _show("connected-error", d.message);
}});

// Render IPA immediately from embedded data (no IPC, no race condition)
(function() {{
  var words = __WORDS_JSON__;
  var err = __INIT_ERROR__;
  if (words) {{
    document.getElementById("ph-content").classList.remove("hidden");
    _renderWords(words);
  }} else if (err) {{
    _show("ph-error", err);
  }}
}})();
</script>
</body>
</html>"""


def _build_html(sentence: str, words_json: str, init_error: str) -> str:
    from .rhythm_view import inject_rhythm_assets

    safe = (
        sentence.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    html = _HTML_TEMPLATE.format(sentence=safe)
    return inject_rhythm_assets(
        html.replace("__WORDS_JSON__", words_json)
        .replace("__INIT_ERROR__", init_error)
        .replace("__SENTENCE_JS__", json.dumps(sentence))
    )


# -- Panel + async orchestration ---------------------------------------------


def _open_panel(wz, sentence: str) -> None:
    if _panel_ref[0] is not None:
        try:
            _panel_ref[0].close()
        except Exception:
            pass

    # Compute IPA synchronously before building HTML — avoids race condition
    from .analyze import analyze_pronunciation_local

    try:
        phonetics = analyze_pronunciation_local(sentence)
        words_json = json.dumps(phonetics["words"], ensure_ascii=False)
        init_error = "null"
    except Exception as e:
        words_json = "null"
        init_error = json.dumps(str(e))

    panel = wz.ui.webview_panel(
        title="Pronunciation",
        html=_build_html(sentence, words_json, init_error),
        width=700,
        height=820,
        floating=True,
    )
    panel.show()
    _panel_ref[0] = panel

    def _on_close():
        if _panel_ref[0] is panel:
            _panel_ref[0] = None

    panel.on_close(_on_close)

    async def _work():
        from .tts import generate_tts

        try:
            audio = await generate_tts(sentence)
            panel.send("audio", audio)
        except Exception as e:
            logger.warning("TTS generation failed: %s", e)
            panel.send("audio_error", {"message": str(e)})

    def _on_request_connected(_data):
        async def _analyze():
            from wenzi.config import load_config

            from .analyze import analyze_pronunciation

            config, _ = load_config()
            try:
                result = await analyze_pronunciation(sentence, config)
                panel.send("connected_speech", result)
            except Exception as e:
                logger.warning("Connected speech analysis failed: %s", e)
                panel.send("connected_error", {"message": str(e)})

        wz.run(_analyze())

    rhythm_request = [None]

    def _on_request_rhythm(data):
        request_id = (data or {}).get("request_id")
        rhythm_request[0] = request_id

        def current():
            return _panel_ref[0] is panel and rhythm_request[0] == request_id

        def send_current(event, payload):
            import threading

            def send():
                if current():
                    panel.send(event, payload)

            if threading.current_thread() is threading.main_thread():
                send()
            else:
                from PyObjCTools import AppHelper

                AppHelper.callAfter(send)

        async def _analyze_rhythm():
            import asyncio

            from .rhythm import analyze_rhythm

            try:
                result = await analyze_rhythm(
                    sentence, should_continue=current,
                    on_progress=lambda done, total: send_current(
                        "rhythm_progress", {"done": done, "total": total, "request_id": request_id}
                    ) if current() else None,
                )
                if current():
                    send_current("rhythm_result", {"result": result, "request_id": request_id})
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if current():
                    send_current("rhythm_error", {"message": str(exc), "request_id": request_id})

        wz.run(_analyze_rhythm())

    def _on_save_audio(data):
        import base64
        import os

        try:
            raw = data["data"].split(",", 1)[1]
            audio_bytes = base64.b64decode(raw)
            path = os.path.join(os.path.expanduser("~/Desktop"), data["filename"])
            with open(path, "wb") as f:
                f.write(audio_bytes)
            logger.info("Audio saved to %s", path)
            panel.send("save_ok", {"path": path})
        except Exception as e:
            logger.warning("Failed to save audio: %s", e)
            panel.send("save_error", {"message": str(e)})

    panel.on("request_connected", _on_request_connected)
    panel.on("request_rhythm", _on_request_rhythm)
    panel.on("save_audio", _on_save_audio)
    wz.run(_work())


# -- Entry point -------------------------------------------------------------


def setup(wz):
    from .dialogue import open_dialogue_panel

    @wz.chooser.source(
        "pronunciation",
        prefix="ph",
        priority=5,
        description="English pronunciation (IPA + TTS)",
        show_preview=False,
        action_hints={"enter": "Analyze"},
    )
    def search(query: str) -> list[dict]:
        q = query.strip()
        if not q:
            return [
                {
                    "title": "Dialogue Reader",
                    "subtitle": "Clipboard, image, or text file",
                    "action": lambda: open_dialogue_panel(wz),
                },
                {
                    "title": "Type an English sentence…",
                    "subtitle": "Get IPA transcription + connected speech + audio",
                }
            ]
        results = [
            {
                "title": q,
                "subtitle": "Enter — analyze pronunciation",
                "action": lambda s=q: _open_panel(wz, s),
            }
        ]
        if "\n" in q or ":" in q or "：" in q:
            results.insert(
                0,
                {
                    "title": "Read as dialogue",
                    "subtitle": "Parse roles and play with different voices",
                    "action": lambda s=q: open_dialogue_panel(wz, s),
                },
            )
        return results
