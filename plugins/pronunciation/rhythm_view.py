"""Shared inline stress and pause styling for both pronunciation readers."""

_CSS = """
:root { --rhythm-stress:#9b3a14; --rhythm-fill:rgba(180,78,28,.10); --rhythm-pause:#68727e; }
@media(prefers-color-scheme:dark) {
  :root { --rhythm-stress:#ffbb87; --rhythm-fill:rgba(255,187,135,.12); --rhythm-pause:#bdc7d2; }
}
.rhythm-stress { color:var(--rhythm-stress); font-weight:750;
  background:var(--rhythm-fill); border-radius:4px; padding:0 .06em; }
.rhythm-word { cursor:help; }
.rhythm-pause { color:var(--rhythm-pause); display:inline-block; padding:0 .24em;
  font-weight:500; font-size:.8em; cursor:help; user-select:none; -webkit-user-select:none; }
.rhythm-legend, .rhythm-summary { font-size:12px; line-height:1.6; color:var(--secondary,var(--muted,#6b7078)); }
.rhythm-legend { margin:10px 0 6px; }
.rhythm-summary { margin:6px 0 12px; }
.rhythm-error { color:#c9372c; font-size:12px; line-height:1.5; }
@media(prefers-color-scheme:dark) { .rhythm-error { color:#ff8a80; } }
"""

_SCRIPT = r"""
var PhRhythm = {
  render: function(container, text, result, offset) {
    offset = offset || 0;
    var stress = new Map((result.stress || []).map(function(item) { return [item.index, item]; }));
    var pauses = new Map((result.pauses || []).map(function(item) { return [item.after, item]; }));
    var parts = text.match(/\s+|\S+/g) || [];
    var fragment = document.createDocumentFragment();
    var index = offset;
    parts.forEach(function(part) {
      if (/^\s+$/.test(part)) { fragment.appendChild(document.createTextNode(part)); return; }
      var hint = stress.get(index);
      var word = document.createElement("span");
      word.textContent = part;
      if (hint) {
        word.className = "rhythm-stress rhythm-word";
        word.title = hint.note;
      }
      fragment.appendChild(word);
      var pause = pauses.get(index);
      if (pause) {
        var mark = document.createElement("span");
        mark.className = "rhythm-pause";
        mark.textContent = pause.kind === "long" ? "||" : "|";
        mark.title = pause.note;
        mark.setAttribute("role", "img");
        mark.setAttribute("aria-label", (pause.kind === "long" ? "Longer pause: " : "Brief pause: ") + pause.note);
        fragment.appendChild(mark);
      }
      index += 1;
    });
    container.replaceChildren(fragment);
  }
};
"""


def inject_rhythm_assets(html: str) -> str:
    return html.replace("</style>", _CSS + "</style>", 1).replace("<script>", "<script>" + _SCRIPT, 1)
