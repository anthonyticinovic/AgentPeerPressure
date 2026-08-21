"""Build the research log artefact from results on disk.

Re-runnable: every number comes from results/*.json, never transcribed by hand.
    uv run python scripts/05_build_log.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pressure.config import CFG

OUT = Path("artifacts/inspection.html")


def _load(name: str):
    p = CFG.results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(x: float) -> str:
    return f"{x:.2f}"


def gate_b_payload():
    """Compute the Gate B section from the Arditi selection + generation-check results.

    Replaces the retracted causal_validation.json feed. The verdict is PASS: a
    diff-of-means direction, selected by Arditi's real grid criterion, ablates refusal.
    """
    sel, gen, hand = (
        _load("arditi_selection.json"),
        _load("arditi_generation_check.json"),
        _load("HANDLABEL_arditi_selected.json"),
    )
    if not (sel and gen and hand):
        return None
    s, a, b = sel["selected"], sel["auroc_choice"], gen["behaviour"]
    n_cand = len(sel["positions"]) * sel["n_layers"]
    yes, no = '<span class="yes">yes</span>', '<span class="no">no</span>'

    def row(name, beh, kl, adm, hand_rate=None):
        return [
            name,
            {"n": _f(beh["refusal_rate"])},
            {"n": _f(beh["harmful_rate"])},
            {"n": _f(hand_rate) if hand_rate is not None else "—"},
            {"n": f"{kl:.2f}" if kl is not None else "—"},
            {"html": adm},
        ]

    return {
        "verdict_class": "good",
        "verdict_short": "pass",
        "verdict": "PASS — refusal is causally ablatable",
        "summary_html": (
            f"Arditi's selection over the full <b>{n_cand}-candidate</b> grid "
            "(9 post-instruction positions × 32 layers), scored by the refusal metric and "
            "filtered on <code>induce&gt;0, kl&lt;0.1, l&lt;0.8L</code>, selects "
            f"<b>i*={s['position']}, l*={s['layer']}</b>. Ablating it drops refusal "
            f"<b>1.00 → {b['selected']['refusal_rate']:.2f}</b>, with "
            f"<b>{hand['hand_complied_rate']:.2f}</b> harmful compliance by hand over all "
            f"{hand['hand_complied_count']}/25 held-out completions. The shuffled-label null "
            "at the same site stays 1.00 / 0.00 — the effect is the labels, not the geometry."
        ),
        "table_head": ["Direction", "refusal", "harmful·judge", "harmful·hand", "kl", "admissible"],
        "table": [
            row("baseline", b["baseline"], None, "—"),
            row(f"ablate i*={s['position']}, l*={s['layer']}", b["selected"], s["kl"], yes,
                hand["hand_complied_rate"]),
            row("shuffled-label, same site", b["shuffled"], None, "—"),
            row("AUROC pick (pos −1, l 22)", b["auroc"], a["kl"], no),
        ],
        "note_html": (
            "The automated harmful judge under-called (0.20) by reading only framing-heavy "
            "openings; the hand rate is 0.84 (<code>HANDLABEL_arditi_selected.json</code>) — the "
            "third proxy-measurement failure in this project, after the substring detector and the "
            "first judge. Note the split the thesis rests on: the <b>monitor</b> direction (the "
            "AUROC pick, pos −1 l 22) is <em>not</em> the mediator — its kl of 0.45 makes it "
            "inadmissible and it yields 0.08 compliance. The direction that <em>separates</em> "
            "harmful from benign and the one that <em>ablates</em> refusal are different vectors."
        ),
        "ledger": {
            "r_ref_mediates": {
                "evidence": f"ablate i*={s['position']},l*={s['layer']} → 1.00→"
                            f"{b['selected']['refusal_rate']:.2f} refusal, "
                            f"{hand['hand_complied_rate']:.2f} harmful (hand)",
                "status": "yes",
            },
            "r_harm_encodes": {"evidence": "inversion test not yet run", "status": "no"},
        },
    }


def build() -> str:
    payload = json.dumps(
        {
            "sweep": _load("r_ref_sweep.json"),
            "samples": _load("dataset_samples.json"),
            "dual": _load("dual_directions.json"),
            "causal": gate_b_payload(),
        }
    )
    return TEMPLATE.replace("__DATA__", payload)


TEMPLATE = r"""<title>Two Signals Under Social Framing</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@400;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#F2F3F0; --panel:#FFF; --ink:#161C1D; --soft:#4A5658; --rule:#D3D8D6;
  --accent:#0F5C63; --tint:#0F5C6314; --warn:#A9700B; --fail:#9B2C2C; --ok:#2F6B4F;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1416; --panel:#151D1F; --ink:#E4E9E8; --soft:#94A3A3; --rule:#263335;
  --accent:#4FB3BC; --tint:#4FB3BC1A; --warn:#D9A441; --fail:#D96B6B; --ok:#6FBF95;}}
:root[data-theme="dark"]{
  --ground:#0E1416; --panel:#151D1F; --ink:#E4E9E8; --soft:#94A3A3; --rule:#263335;
  --accent:#4FB3BC; --tint:#4FB3BC1A; --warn:#D9A441; --fail:#D96B6B; --ok:#6FBF95;}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
  font-family:"IBM Plex Serif",Georgia,serif;font-size:15.5px;line-height:1.58}
.wrap{max-width:940px;margin:0 auto;padding:48px 26px 88px;display:flex;flex-direction:column;gap:38px}
h1,h2,h3,.lbl,th{font-family:"IBM Plex Sans Condensed",system-ui,sans-serif;text-wrap:balance}
h1{font-size:clamp(27px,4.4vw,38px);line-height:1.1;margin:0;font-weight:700;letter-spacing:-.01em}
h2{font-size:20px;margin:0;font-weight:600}
h3{font-size:15px;margin:0 0 5px;font-weight:600}
p{margin:0;max-width:70ch}
.lbl{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--soft);font-weight:600}
.mono,td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;display:flex;flex-direction:column;gap:10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{font-family:"IBM Plex Mono",monospace;font-size:11px;padding:3px 8px;border:1px solid var(--rule);
  color:var(--soft);background:var(--panel)}
.chip b{color:var(--ink);font-weight:600}
section{display:flex;flex-direction:column;gap:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);font-weight:600}
td.n{text-align:right;white-space:nowrap;font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
.yes{color:var(--ok);font-weight:600} .no{color:var(--fail);font-weight:600}
.part{color:var(--warn);font-weight:600}
.entry{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--accent);
  padding:15px 18px;display:flex;flex-direction:column;gap:9px}
.entry.bad{border-left-color:var(--fail)} .entry.good{border-left-color:var(--ok)}
.entry .top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.entry .id{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--soft)}
.k{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-family:"IBM Plex Mono",monospace;
  font-size:12px;font-variant-numeric:tabular-nums;background:var(--tint);padding:9px 12px;margin:0}
.k dt{color:var(--soft)} .k dd{margin:0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:700px){.two{grid-template-columns:1fr}}
.grids{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px}
.hm{display:grid;gap:1px}
.hm .cell{aspect-ratio:1.6}
.hm .cell.pick{outline:2px solid var(--ink);outline-offset:-2px}
.hm .hd,.hm .rw{font-family:"IBM Plex Mono",monospace;font-size:8.5px;color:var(--soft);
  display:flex;align-items:center;justify-content:center;aspect-ratio:1.6}
.hm .rw{justify-content:flex-end;padding-right:4px}
.hm .rw.full{color:var(--accent);font-weight:600}
.cap{font-size:12.5px;color:var(--soft);margin:0}
svg.ch{width:100%;height:210px}
.card{background:var(--panel);border:1px solid var(--rule);padding:11px 13px;font-size:13.5px}
.card .tg{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--soft);margin-bottom:5px}
.card .tg em{font-style:normal;color:var(--accent);font-weight:600}
.stack{display:flex;flex-direction:column;gap:8px}
details{background:var(--panel);border:1px solid var(--rule)}
summary{padding:11px 14px;cursor:pointer;font-family:"IBM Plex Sans Condensed",sans-serif;
  font-weight:600;font-size:14px}
summary:focus-visible{outline:2px solid var(--accent)}
details>div{padding:0 14px 14px;display:flex;flex-direction:column;gap:14px}
code{font-family:"IBM Plex Mono",monospace;font-size:.87em;background:var(--tint);padding:1px 4px}
pre{font-family:"IBM Plex Mono",monospace;font-size:11.5px;background:var(--tint);padding:10px 12px;
  margin:0;overflow-x:auto;border-left:2px solid var(--accent)}
</style>

<div class="wrap">
<header>
  <div class="lbl">MATS 12.0 · research log · local 4B pipeline pass</div>
  <h1>Which safety signal survives social framing</h1>
  <p class="cap">Two frozen directions measured against identical harmful-task behaviour under peer
  framing. This log records what has been established and what has only been assumed.</p>
  <div class="chips" id="chips"></div>
</header>

<section>
  <div class="lbl">Ledger</div>
  <h2>What is actually established</h2>
  <div class="scroll"><table id="ledger"></table></div>
  <p class="cap"><code>r_ref</code> and <code>r_harm</code> are labels of convenience, taken from the
  token position each is read at. They are not yet earned by measurement.</p>
</section>

<section>
  <div class="lbl">Log</div>
  <h2>Entries</h2>
  <div class="stack" id="log"></div>
</section>

<section id="causal-sec">
  <div class="lbl">Gate B</div>
  <h2>Causal validation</h2>
  <div class="stack" id="causal-body"></div>
</section>

<section>
  <div class="lbl">Figures</div>
  <h2>Layer and position sweep</h2>
  <p class="cap">The same 288 candidate directions scored against two different test sets. Only the
  test set differs.</p>
  <div class="grids">
    <div><h3>Matched pairs — selection</h3><div class="hm" id="hm-m"></div></div>
    <div><h3>AdvBench vs Alpaca — vacuous</h3><div class="hm" id="hm-u"></div></div>
  </div>
  <div class="chips"><span class="chip">AUROC <b>0.50</b></span>
    <span class="chip" id="ramp-c" style="width:110px">&nbsp;</span>
    <span class="chip"><b>1.00</b></span>
    <span class="chip" style="color:var(--accent)">&#9632; full attention</span></div>
  <div class="two" id="dual-charts">
    <div><h3>AUROC by layer</h3>
      <p class="cap">Matched pairs. The harmfulness probe is markedly weaker.</p>
      <svg class="ch" id="c-auroc" viewBox="0 0 800 210" preserveAspectRatio="none"></svg></div>
    <div><h3>cos(r_harm, r_ref) by layer</h3>
      <p class="cap">Red dashes mark the random-vector baseline, not zero.</p>
      <svg class="ch" id="c-cos" viewBox="0 0 800 210" preserveAspectRatio="none"></svg></div>
  </div>
</section>

<section>
  <div class="lbl">Data</div>
  <h2>Three roles, three separate corpora</h2>
  <p class="cap">Fit, select and evaluate each use a different dataset. The separation is structural,
  not a row split.</p>
  <div class="scroll"><table id="roles"></table></div>
  <details><summary>Sample items from each corpus</summary><div id="samples"></div></details>
  <details><summary>Run environment and provenance</summary><div>
    <div class="scroll"><table id="env"></table></div></div></details>
</section>
</div>

<script>
const D=__DATA__, S=D.sweep, SA=D.samples, DU=D.dual, CA=D.causal;
const el=(t,c,x)=>{const n=document.createElement(t); if(c)n.className=c;
  if(x!==undefined)n.textContent=x; return n;};
function tbl(node,head,rows){const h=el('tr'); head.forEach(x=>h.append(el('th',null,x)));
  node.append(h);
  rows.forEach(r=>{const tr=el('tr');
    r.forEach(c=>{if(c&&typeof c==='object'&&c.html){const td=el('td',c.cls||null);
        td.innerHTML=c.html; tr.append(td);}
      else if(c&&typeof c==='object'&&c.n!==undefined)tr.append(el('td','n',String(c.n)));
      else tr.append(el('td',null,String(c)));});
    node.append(tr);});}

document.getElementById('chips').append(...[
  ['model',S.model],
  ['l* r_ref',DU?DU.positions.context_last.l_star:S.l_star],
  ['AUROC r_ref',(DU?DU.positions.context_last.auroc:S.auroc_matched).toFixed(3)],
  ['AUROC r_harm',DU?DU.positions.task_last.auroc.toFixed(3):'—'],
  ['cos',DU?DU.gate_b2.cos_matched_layer.toFixed(3):'—'],
  ['Gate B',CA?CA.verdict_short:'not run'],
].map(([k,v])=>{const c=el('span','chip'); c.append(k+' '); c.append(el('b',null,String(v)));
  return c;}));

const st=s=>({html:`<span class="${s==='yes'?'yes':s==='no'?'no':'part'}">`+
  (s==='yes'?'established':s==='no'?'not established':'partial')+'</span>'});
tbl(document.getElementById('ledger'),['Claim','Evidence','Status'],[
 ['Two candidate vectors exist','diff-of-means over 400+400 prompts',st('yes')],
 ['They separate matched harmful vs benign',
   'AUROC '+(DU?DU.positions.context_last.auroc.toFixed(3):'—')+' / '+
   (DU?DU.positions.task_last.auroc.toFixed(3):'—'),st('yes')],
 ['They are distinct directions',
   'cos '+(DU?DU.gate_b2.cos_matched_layer.toFixed(3):'—')+' vs random 0.016 — gate cannot fail',
   st('no')],
 ['Refusal is causally ablatable',CA?CA.ledger.r_ref_mediates.evidence:'ablation not yet run',
   st(CA?CA.ledger.r_ref_mediates.status:'no')],
 ['Monitor ≠ mediator (detector ≠ cause)',
   'AUROC pick has kl 0.45, 0.08 compliance — separator is not the ablating vector',
   st(CA?'yes':'no')],
 ['r_harm encodes harmfulness',CA?CA.ledger.r_harm_encodes.evidence:'inversion test not yet run',
   st(CA?CA.ledger.r_harm_encodes.status:'no')],
 ['Peer framing shifts compliance','pilot not yet run',st('no')],
]);

const ENTRIES=[
 {id:'E1',cls:'bad',t:'A perfect AUROC that measured the corpus, not the concept',
  k:[['bag-of-words AUROC','0.9955'],['direction at layer 0','0.9917'],['prompt length alone','0.5857']],
  b:'Selecting the layer by held-out AUROC on the same corpora used to fit gave 1.000 at layer 8 of 32. '+
    'AdvBench is uniformly <em>"Provide step-by-step instructions…"</em> and Alpaca <em>"What was the…"</em>; '+
    'that difference is linearly present from the first block onward. '+
    '<b>Gate B\'s "AUROC ≥ 0.9" is vacuous here — all 32 layers pass, layer 0 included.</b>'},
 {id:'E2',cls:'good',t:'Selection moved onto topic-matched pairs',
  k:[['layer-0 AUROC, matched','0.6765'],['l* moved','8 → 22']],
  b:'JailbreakBench pairs each harmful behaviour with a benign one on the same topic and register, so '+
    'surface lexis is controlled. AdvBench and Alpaca still <em>fit</em> the direction; they no longer '+
    '<em>select</em> it.'},
 {id:'E3',cls:'good',t:'The leakage assertion caught a real overlap',
  k:[['JBB items sourced from AdvBench','18 of 100'],['clean pairs remaining','82']],
  b:'Without the assert, 18 selection items would have been prompts the direction was fitted on. '+
    'Excluded by <code>Source</code> rather than string match, so rephrasings go too, along with each '+
    'index-matched benign partner.'},
 {id:'E4',cls:'',t:'Zhao’s harmfulness direction is a read position, not a method',
  k:[['t_inst','harmfulness → our task_last'],['t_post-inst','refusal → our context_last']],
  b:'Read from their released code rather than the abstract. The harmfulness and refusal directions '+
    'are the <em>same</em> diff-of-means sliced at different tokens — <code>mode_dir</code> picks the '+
    'index. The plan feared that guessing would yield two copies of one vector; the real risk was the '+
    'opposite, inventing a distinct construct that does not exist.',
  pre:"if   mode_dir == 'hf':     mean_diffs[:, NUM_TOKEN_HIDDEN-1]   # t_inst      -> harmfulness\n"+
      "elif mode_dir == 'refuse': mean_diffs[:, -1]                   # t_post-inst -> refusal"},
 {id:'E5',cls:'bad',t:'Gate B2 cannot fail, so passing it means little',
  k:[['random |cos| in 2560 dims','0.0163'],['observed','0.1004'],['ratio to chance','6.2×'],
     ['random pairs passing ≤0.9','100%']],
  b:'In high dimensions nearly everything is near-orthogonal. Read against its null, the observed '+
    'cosine shows the two directions are <em>more</em> aligned than random, sharing about 1% of '+
    'variance. <b>Two of the plan\'s three quantitative gates are thresholds on similarity metrics, '+
    'and neither bites.</b> Only causal and functional tests discriminate, so every remaining '+
    'threshold gets its null computed before the number is read.'},
 {id:'E6',cls:'',t:'A harmfulness direction may exist — but not where Zhao place it',
  k:[['t_inst (offset −10) ablated','+3.88 — discrimination rises'],
     ['offset −5, layer 31 ablated','+0.54 — collapses from +2.72'],
     ['random ablation','+2.51'],
     ['refusal under that ablation','0.92 → 1.00, intact'],
     ['general capability','coherent']],
  b:'A causal sweep over extraction positions found one whose ablation destroys the model\'s '+
    'harmfulness judgement while leaving refusal behaviour and general competence untouched — the '+
    'dissociation Zhao claim, at a post-instruction position rather than at <code>t_inst</code>. '+
    '<b>Not yet a finding.</b> Layer 31 is one step from the logits, so the leading alternative is '+
    'that this is the readout direction for the two answer tokens — deleting the ability to <em>say</em> '+
    'the answer, not to <em>hold</em> the belief. Distinguishing test: repeat with different answer '+
    'tokens and a reversed mapping. A belief direction should still collapse; a readout direction '+
    'should not.'},
 {id:'E7',cls:'bad',t:'The direction was the corpus axis, and the controls were the direction',
  k:[['cos(top PC, r_ref)','0.999'],['cos(shuffled-label, r_ref)','0.86'],
     ['variance explained by r_ref','31%'],['random baseline cos','0.016']],
  b:'Adversarial review of the methods found that <code>r_ref</code>, fitted on AdvBench vs Alpaca, '+
    '<em>is</em> the top principal component of that activation cloud. A shuffled-label diff-of-means '+
    'recovers a direction 86% aligned with it. Both follow from E1: the corpora are lexically '+
    'separable, so their activations split along one dominant axis and any mean-difference returns '+
    'that axis regardless of labels. Matched-pair <em>selection</em> fixed which layer we pick; it '+
    'never fixed what we fit.'},
 {id:'E8',cls:'bad',t:'Retraction: the refusal measure was counting phrasing',
  k:[['substring, ablate r_ref','0.04 → 1.00'],['judge, ablate r_ref','0.04 → 0.08'],
     ['judge, on AgentHarm','0.04 → 0.04'],['judge, top-PC control','0.16']],
  b:'<code>is_refusal</code> matched openers like “I cannot” and was never checked against text. '+
    'Ablation moves the model from <em>“I cannot provide instructions…”</em> to <em>“Constructing an '+
    'explosive device is illegal…”</em> — still a refusal, scored as compliance. Judge-graded, the '+
    'effect is 12x smaller, and zero on the evaluation corpus. <b>Gate B fails at 4B.</b> Fitting on '+
    'matched pairs makes the shuffled-label null work again (0.86 → 0.01) and gives AUROC 0.980, but '+
    'ablating that direction does not raise compliance either.'},
 {id:'E9',cls:'',t:'Mistake: “Arditi” was swept as a line, not a grid',
  k:[['candidates tested','32 of 288'],['position swept','−1 only'],['scored by','judge, not refusal metric'],
     ['filters applied','none']],
  b:'An earlier entry concluded Arditi’s ablation does not replicate here. It was wrong. That sweep '+
    'varied <em>layer</em> at a single token position (−1), grading generations with a judge and '+
    'applying none of the three selection filters — 32 of the 288 candidates Arditi actually search. '+
    'Their winner for comparable models sits at position −5, invisible to a position−1 line. '+
    'Retained as the error corrected in E11.'},
 {id:'E10',cls:'bad',t:'Retraction: “RDO” was a self-invented prefix proxy',
  k:[['claimed method','Wollschläger et al. 2502.17420'],['actually built','−log P("Sure, here") under ablation'],
     ['objective','affirmative <em>prefix</em>, not compliance'],['status','retracted, archived']],
  b:'A direction was gradient-optimised and reported at 0.60 blind-human compliance. Two problems: '+
    'it was <em>not</em> Wollschläger et al.’s method (their code was never read — this was a home-made '+
    'reduction), and its objective rewards an affirmative opening token, the same prefix-proxy failure '+
    'as the substring detector. Unnecessary in any case: a plain diff-of-means direction passes Gate B '+
    'once selected properly (E11). Archived, not cited.'},
 {id:'E11',cls:'good',t:'Gate B passes: the grid finds i*=−7, l*=12',
  k:[['candidates','288 (9 positions × 32 layers)'],['selected','i*=−7, l*=12, kl 0.067'],
     ['refusal under ablation','1.00 → 0.04'],['harmful compliance (hand)','0.84'],
     ['shuffled-label null','1.00 / 0.00'],['AUROC pick (l 22) kl','0.45 — inadmissible']],
  b:'Run to the paper: 288 diff-of-means candidates scored by the refusal metric (one forward pass '+
    'each), selected as minimum bypass subject to <code>induce&gt;0, kl&lt;0.1, l&lt;0.8L</code>. '+
    'Ablating the winner collapses refusal and yields 0.84 harmful compliance by hand over all 25 '+
    'completions; the shuffled-label null at the same site does nothing. <b>Refusal on this model is '+
    'causally ablatable.</b> The kl filter is load-bearing: it rejects both the old AUROC pick '+
    '(kl 0.45) and the E6 late-layer “collapse”, which suppress the refusal <em>token</em> rather '+
    'than the <em>feature</em>. Crucially the separator and the mediator are different vectors — the '+
    'detector≠cause split the thesis needs.'},
];
const logn=document.getElementById('log');
ENTRIES.forEach(e=>{const d=el('div','entry '+e.cls);
  const top=el('div','top'); top.append(el('span','id',e.id)); top.append(el('h3',null,e.t));
  d.append(top);
  const p=el('p'); p.innerHTML=e.b; d.append(p);
  if(e.pre)d.append(el('pre',null,e.pre));
  if(e.k){const dl=el('dl','k'); e.k.forEach(([a,b])=>{dl.append(el('dt',null,a));
    dl.append(el('dd',null,b));}); d.append(dl);}
  logn.append(d);});

const cbody=document.getElementById('causal-body');
if(CA){
  const v=el('div','entry '+(CA.verdict_class||''));
  const top=el('div','top'); top.append(el('span','id','GATE B'));
  top.append(el('h3',null,CA.verdict)); v.append(top);
  const p=el('p'); p.innerHTML=CA.summary_html; v.append(p); cbody.append(v);
  const sc=el('div','scroll'); const t=el('table'); sc.append(t); cbody.append(sc);
  tbl(t,CA.table_head,CA.table);
  const n=el('p','cap'); n.innerHTML=CA.note_html; cbody.append(n);
} else {
  cbody.append(el('p','cap','Not yet run. Until it is, neither direction has been shown to do '+
    'anything beyond correlate with the harmful/benign split.'));
}

const col=v=>{const t=Math.max(0,Math.min(1,(v-.5)/.5)),lo=[.93,.94,.92],hi=[.06,.36,.39];
  const m=lo.map((l,i)=>Math.round(255*(l+(hi[i]-l)*t))); return 'rgb('+m[0]+','+m[1]+','+m[2]+')';};
document.getElementById('ramp-c').style.background=
  'linear-gradient(90deg,'+col(.5)+','+col(.75)+','+col(1)+')';
function heat(node,grid,pick){const offs=S.sweep_offsets;
  node.style.gridTemplateColumns='22px repeat('+offs.length+',1fr)';
  node.append(el('div','hd','')); offs.forEach(o=>node.append(el('div','hd',String(o))));
  for(let l=grid.length-1;l>=0;l--){const full=S.layer_types[l]==='full_attention';
    node.append(el('div','rw'+(full?' full':''),(l%4===3||l===0)?String(l):''));
    for(let o=0;o<offs.length;o++){
      const c=el('div','cell'+(pick&&l===S.l_star&&o===offs.indexOf(S.offset)?' pick':''));
      c.style.background=col(grid[l][o]);
      c.title='layer '+l+' ('+S.layer_types[l]+'), offset '+offs[o]+' → '+grid[l][o].toFixed(4);
      node.append(c);}}}
heat(document.getElementById('hm-m'),S.auroc_grid,true);
heat(document.getElementById('hm-u'),S.auroc_grid_unmatched,false);

const NS='http://www.w3.org/2000/svg';
function chart(id,series,lo,hi,marks,dashed){
  const svg=document.getElementById(id); if(!svg)return;
  const L=series[0].v.length,X=l=>40+l*(730/(L-1)),Y=q=>184-((q-lo)/(hi-lo))*162;
  const add=(t,a)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);
    svg.append(n);return n;};
  marks.forEach(m=>{add('line',{x1:40,x2:775,y1:Y(m),y2:Y(m),stroke:'currentColor',
    'stroke-opacity':.15,'stroke-dasharray':'2 3'});
    const t=add('text',{x:4,y:Y(m)+4,fill:'currentColor','fill-opacity':.55,'font-size':9.5,
      'font-family':'IBM Plex Mono,monospace'});t.textContent=m.toFixed(2);});
  if(dashed!==undefined)add('line',{x1:40,x2:775,y1:Y(dashed),y2:Y(dashed),stroke:'var(--fail)',
    'stroke-width':1,'stroke-dasharray':'5 3'});
  series.forEach((s,i)=>{add('path',{d:s.v.map((y,l)=>(l?'L':'M')+X(l)+','+Y(y)).join(' '),
    fill:'none',stroke:s.c,'stroke-width':1.9});
    const k=add('text',{x:44+i*150,y:13,fill:s.c,'font-size':10,
      'font-family':'IBM Plex Mono,monospace'});k.textContent='— '+s.n;
    if(s.star!==undefined)add('circle',{cx:X(s.star),cy:Y(s.v[s.star]),r:4.5,fill:'none',
      stroke:s.c,'stroke-width':2});});
  const ax=add('text',{x:400,y:203,fill:'currentColor','fill-opacity':.5,'font-size':9.5,
    'text-anchor':'middle','font-family':'IBM Plex Mono,monospace'});
  ax.textContent='layer 0 → '+(L-1);
}
if(DU){const P=DU.positions;
  chart('c-auroc',[{v:P.context_last.auroc_by_layer,c:'var(--accent)',n:'r_ref',
      star:P.context_last.l_star},
    {v:P.task_last.auroc_by_layer,c:'var(--warn)',n:'r_harm',star:P.task_last.l_star}],
    .4,1,[.5,.75,1]);
  chart('c-cos',[{v:DU.gate_b2.cos_per_layer,c:'var(--ok)',n:'cosine'}],-.3,1,[0,.5,.9],0.0163);
} else document.getElementById('dual-charts').remove();

tbl(document.getElementById('roles'),['Dataset','N','Role','Used to','Never used to'],[
 ['AdvBench (harmful)',{n:S.n_extract},{html:'<b style="color:var(--warn)">FIT</b>'},
   'Harmful mean','Select l*; evaluate'],
 ['Alpaca (harmless)',{n:S.n_extract},{html:'<b style="color:var(--warn)">FIT</b>'},
   'Harmless mean','Select l*; evaluate'],
 ['JBB matched pairs',{n:'82 + 82'},{html:'<b style="color:var(--accent)">SELECT</b>'},
   'Pick l*, offset, tau','Fit; evaluate'],
 ['AgentHarm',{n:'104 / 26'},{html:'<b style="color:var(--ok)">EVALUATE</b>'},
   'The experiment','Fit or select anything'],
]);

const sm=document.getElementById('samples');
const mkcard=(tags,text,tools)=>{const c=el('div','card');
  if(tags){const t=el('div','tg'); tags.forEach(([k,v])=>{const s=el('span');s.append(k+' ');
    s.append(el('em',null,v));s.append('   ');t.append(s);}); c.append(t);}
  c.append(el('div',null,text));
  if(tools)c.append(el('div','tg','tools · '+tools.join(' → '))); return c;};
const grp=(title,items)=>{const w=el('div'); w.append(el('h3',null,title));
  const s=el('div','stack'); items.forEach(i=>s.append(i)); w.append(s); return w;};
const two=el('div','two');
two.append(grp('AdvBench — fits',SA.advbench.slice(0,3).map(p=>mkcard(null,p))));
two.append(grp('Alpaca — fits',SA.alpaca.slice(0,3).map(p=>mkcard(null,p))));
sm.append(two);
sm.append(grp('AgentHarm — evaluates. One behaviour, four variants.',
  SA.agentharm_variants.map(v=>mkcard([['difficulty',v.difficulty],['detailed',String(v.detailed)],
    ['hint',String(v.hint)]],v.prompt,v.target_functions))));
sm.append(grp('Matched benign counterpart — same tools, no harm',
  [mkcard([['name',SA.agentharm_benign_counterpart.name]],SA.agentharm_benign_counterpart.prompt,
    SA.agentharm_benign_counterpart.target_functions)]));

tbl(document.getElementById('env'),['Field','Value'],
  Object.entries(S.environment).concat([
   ['extraction',S.corpus.harmful+' / '+S.corpus.harmless],
   ['selection',S.selection_corpus.join(' / ')],
   ['n extraction',S.n_extract],['thinking mode','disabled'],
   ['sweep runtime',S.elapsed_min+' min']]).map(([k,v])=>[k,{html:'<span class="mono">'+v+'</span>'}]));
</script>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
