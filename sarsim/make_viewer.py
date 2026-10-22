"""Build a self-contained interactive HTML viewer (viz/viewer.html) from simulation traces."""
import json, sys
from .visualize import load_params, record, states
CODE = {"unvisited": 0, "sensing": 1, "rejected": 2, "tp": 3, "fp": 4}
CASES = [("D", 74, ("static", "active"), "Scenario D: distractors, full battery"),
         ("E", 14, ("static_RB", "active"), "Scenario E: 30% battery"),
         ("C", 0, ("active", "active_mt"), "Scenario C: three victims, PIR gate versus mandatory temperature")]
NAMES = {"static": "Static fusion", "active": "Active (proposed)", "active_mt": "Active + mandatory temperature", "static_RB": "Static + reserve + battery weights"}

def build(out="viz/viewer.html"):
    P = load_params(); data = []
    for scen, ep, pols, title in CASES:
        runs = []
        for p in pols:
            lay, tr, res = record(scen, ep, p, P); fr = states(tr, lay)
            runs.append(dict(name=NAMES[p], lay={k: lay[k] for k in ("xy", "victim", "hazard", "dheat", "dmotion", "obst", "batt0")},
                             f=[[round(f["xy"][0], 2), round(f["xy"][1], 2), [CODE[s] for s in f["status"]], f["msg"], round(f["batt"], 1), round(f["energy"], 1), f["act"], f["tp"], f["fp"]] for f in fr],
                             res=dict(tp=res["tp"], n=res["n_victims"], fp=res["fp"], energy=res["energy"], act=res["activations"], ok=bool(res["success"]) if scen == "E" else None)))
        data.append(dict(title=f"{title} (episode {ep})", runs=runs))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(out, "w", encoding="utf8").write(html); print(out, len(html) // 1024, "KB")

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Active, risk-aware SAR search: simulation replay</title>
<style>
:root{--bg:#fff;--fg:#1c1c1c;--mut:#666;--card:#f4f4f2;--line:#bbb;--ok:#1b9e3b;--bad:#d62728;--acc:#2166ac;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#ececec;--mut:#9a9a9a;--card:#1f2226;--line:#444}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#ececec;--mut:#9a9a9a;--card:#1f2226;--line:#444}
html,body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 Georgia,"Times New Roman",serif}
main{max-width:980px;margin:0 auto;padding:14px}h1{font-size:19px;margin:6px 0}p.n{color:var(--mut);font-size:13px;margin:4px 0 10px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}button,select{font:inherit;padding:6px 11px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px}
button.on{border-color:var(--acc);outline:1px solid var(--acc)}input[type=range]{flex:1;min-width:160px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:640px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:8px;padding:8px}.card h2{font-size:15px;margin:2px 0 4px}canvas{width:100%;aspect-ratio:1;display:block}
.st{font-size:13px}.leg{font-size:12px;color:var(--mut)}</style></head><body><main>
<h1>Active, risk-aware multimodal search: simulation replay</h1>
<p class="n">Illustrative episodes from the released simulation (assumed sensor models, not hardware). Episodes were chosen to show differences between policies and are not typical; the paper reports averages over 300 episodes per scenario.</p>
<div class="row" id="cases"></div>
<div class="row"><button id="play">Pause</button><button id="rst">Restart</button><label>Speed <select id="spd"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><input type="range" id="sl" min="0" value="0"></div>
<div class="grid"><div class="card"><h2 id="n0"></h2><canvas id="c0" width="520" height="520"></canvas><div class="st" id="s0"></div></div>
<div class="card"><h2 id="n1"></h2><canvas id="c1" width="520" height="520"></canvas><div class="st" id="s1"></div></div></div>
<p class="leg">Pink dot: robot. Grey: visited and rejected. Green: confirmed victim. Red: false alarm. Yellow: being sampled. Outline markers show hidden truth: ★ victim, ▲ hazard, ◆ distractor, □ obstacle.</p>
</main><script>
const D=__DATA__;let ci=0,k=0,play=true,last=0;const $=i=>document.getElementById(i),cv=[$("c0"),$("c1")];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim(),SC=["#c8c8c8","#f2c200","#8a8a8a","#1b9e3b","#d62728"];
function maxN(){return Math.max(...D[ci].runs.map(r=>r.f.length))+12}
function draw(c,run,kk){const g=c.getContext("2d"),W=c.width,s=W/24,o=W/2,f=run.f[Math.min(kk,run.f.length-1)],L=run.lay;g.clearRect(0,0,W,W);
const X=x=>o+x*s,Y=y=>o-y*s;g.strokeStyle=css("--line");g.setLineDash([4,4]);g.beginPath();g.arc(o,o,10*s,0,7);g.stroke();g.setLineDash([]);
g.strokeStyle="#4c72b0";g.globalAlpha=.7;g.beginPath();for(let i=0;i<=Math.min(kk,run.f.length-1);i++){const p=run.f[i];i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1]))}g.stroke();g.globalAlpha=1;
L.xy.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);if(L.obst[i]){g.strokeStyle=css("--fg");g.strokeRect(x-14,y-14,28,28)}
g.fillStyle=SC[f[2][i]];g.beginPath();g.arc(x,y,10,0,7);g.fill();g.strokeStyle=css("--fg");g.lineWidth=1;g.stroke();g.font="15px sans-serif";g.textAlign="center";g.textBaseline="middle";g.fillStyle="#000";
let t="";if(L.victim[i])t+="★";if(L.hazard[i])t+="▲";if(L.dheat[i]||L.dmotion[i])t+="◆";g.fillText(t,x,y+1)});
g.fillStyle=css("--fg");g.fillRect(o-5,o-5,10,10);g.fillStyle="#e377c2";g.beginPath();g.arc(X(f[0]),Y(f[1]),8,0,7);g.fill();g.strokeStyle=css("--fg");g.stroke();
const fr=Math.max(f[4],0)/L.batt0;g.strokeRect(20,W-28,W-40,12);g.fillStyle=fr>.2?"#1b9e3b":"#d62728";g.fillRect(20,W-28,(W-40)*fr,12);return f}
function render(){const m=maxN();$("sl").max=m;$("sl").value=k;D[ci].runs.forEach((r,j)=>{const f=draw(cv[j],r,Math.floor(k));$("n"+j).textContent=r.name;
$("s"+j).innerHTML=f[3]+"<br>victims found "+f[7]+" of "+r.res.n+" · false alarms "+f[8]+"<br>energy used "+Math.round(f[5])+" · sensor samples "+f[6]+(Math.floor(k)>=r.f.length-1&&r.res.ok!==null?"<br>mission "+(r.res.ok?"success":"failed"):"")})}
function tick(t){if(play){if(t-last>60){k+=+$("spd").value*(t-last)/60>0?(+$("spd").value):0;last=t;if(k>=maxN())k=maxN()}render()}requestAnimationFrame(tick)}
function sel(i){ci=i;k=0;[...$("cases").children].forEach((b,j)=>b.classList.toggle("on",j==i));render()}
D.forEach((d,i)=>{const b=document.createElement("button");b.textContent=d.title;b.onclick=()=>sel(i);$("cases").appendChild(b)});
$("play").onclick=()=>{play=!play;$("play").textContent=play?"Pause":"Play"};$("rst").onclick=()=>{k=0};$("sl").oninput=e=>{k=+e.target.value;render()};sel(0);requestAnimationFrame(tick);
</script></body></html>"""
if __name__ == "__main__": build(*(sys.argv[1:2]))
