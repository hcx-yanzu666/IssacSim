"""Build a self-contained HTML evaluation report from YOLO pick run folders."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "yolo_pick"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation_report.html"
DEFAULT_EXCLUSIONS = ROOT / "configs" / "evaluation_exclusions.json"


def millimeters(value):
    return float(value) * 1000 if isinstance(value, (int, float)) and math.isfinite(value) else None


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _relative_file(path, output_dir):
    if not path.is_file():
        return None
    try:
        return urllib.parse.quote(Path(os.path.relpath(path, output_dir)).as_posix())
    except ValueError:
        # Windows cannot form a relative path between different drives.
        return path.resolve().as_uri()


def load_runs(run_root, output_dir, excluded):
    runs = []
    ignored = []
    for result_path in sorted(run_root.glob("*/result.json")):
        outer = _read_json(result_path)
        if not isinstance(outer, dict):
            ignored.append(result_path.parent.name)
            continue
        run_id = str(outer.get("run_id") or result_path.parent.name)
        if run_id in excluded:
            ignored.append(run_id)
            continue

        inner = outer.get("grasp_result")
        if not isinstance(inner, dict):
            inner = {}
        perception = outer.get("perception") or inner.get("perception") or {}
        yolo_metadata = perception.get("yolo") or {}
        status = str(inner.get("status") or outer.get("status") or "Unknown")
        phase = str(inner.get("phase") or outer.get("phase") or "Unknown")
        metrics = inner.get("metrics") or {}
        estimate = _read_json(result_path.parent / "bottle_pose_estimate.json") or {}
        yolo = _read_json(result_path.parent / "yolo_result.json") or {}
        files = {}
        for key, name in (("rgb", "rgb.png"), ("mask", "yolo_mask.png"),
                          ("prediction", "yolo_prediction.png")):
            files[key] = _relative_file(result_path.parent / name, output_dir)

        runs.append({
            "run_id": run_id,
            "started_at": outer.get("started_at"),
            "status": status,
            "phase": phase,
            "detail": inner.get("detail") or outer.get("detail") or "",
            "confidence": yolo_metadata.get("confidence", yolo.get("confidence")),
            "pick_verified": inner.get("pick_verified"),
            "release_commanded": inner.get("release_commanded"),
            "placement_error_mm": millimeters(metrics.get("xy_error_m")),
            "slip_mm": millimeters(inner.get("max_hold_transport_slip_m")),
            "pose_error_mm": millimeters(estimate.get("translation_error_m_evaluation_only")),
            "max_lift_mm": millimeters(inner.get("max_lift_m")),
            "sim_seconds": inner.get("elapsed_seconds"),
            "wall_seconds": outer.get("wall_seconds"),
            "files": files,
            "result_file": _relative_file(result_path, output_dir),
            "configuration": inner.get("configuration") or {},
            "events": outer.get("events") or [],
        })

    runs.sort(key=lambda run: (str(run.get("started_at") or ""), run["run_id"]))
    return runs, ignored


def render_html(runs, ignored, run_root):
    acceptance = _read_json(ROOT / "configs/task_acceptance.json") or {}
    payload = json.dumps({
        "runs": runs,
        "ignored_run_ids": ignored,
        "source": str(run_root),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "placement_limit_mm": acceptance.get("placement_xy_error_m", 0.020) * 1000,
        "slip_limit_mm": acceptance.get("max_transport_slip_m", 0.025) * 1000,
    }, ensure_ascii=False, separators=(",", ":"))
    payload = (payload.replace("&", "\\u0026").replace("<", "\\u003c")
              .replace(">", "\\u003e").replace("\u2028", "\\u2028")
              .replace("\u2029", "\\u2029"))

    return r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>机器人抓取评测报告</title>
<style>
:root{color-scheme:light;--ink:#202b2b;--muted:#647372;--line:#dbe3df;--paper:#f5f7f4;--white:#fff;--green:#147d69;--green-soft:#e5f3ee;--orange:#bf642f;--orange-soft:#fcf0e8;--red:#b5463f;--red-soft:#fbecea;--blue:#477b9c;--blue-soft:#eaf2f7;--shadow:0 12px 34px rgba(25,50,43,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,"Segoe UI",Arial,"Microsoft YaHei",sans-serif}button,select{font:inherit;color:inherit}button{cursor:pointer}.shell{max-width:1320px;margin:auto;padding:26px 34px 52px}.topline{display:flex;align-items:center;justify-content:space-between;padding:0 0 17px;border-bottom:1px solid var(--line);color:var(--muted);font-size:12px}.brand{display:flex;align-items:center;gap:10px;font-weight:700;color:var(--ink)}.brand-mark{width:11px;height:11px;background:var(--green);border-radius:2px;box-shadow:7px 0 0 #8fc8b7}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:30px;padding:36px 0 29px}.eyebrow{text-transform:uppercase;font-size:11px;font-weight:750;color:var(--green);letter-spacing:0}.hero h1{font-size:30px;line-height:1.15;margin:8px 0 9px;font-weight:720}.hero p{margin:0;color:var(--muted);max-width:680px}.generated{text-align:right;color:var(--muted);font-size:12px;white-space:nowrap}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#edf2ee;border:1px solid var(--line);padding:10px 12px;border-radius:6px;margin-bottom:17px}.toolbar label{font-size:12px;color:var(--muted);font-weight:650}.toolbar select{min-width:185px;border:1px solid #cbd6d1;background:white;border-radius:4px;padding:7px 30px 7px 9px}.toolbar-note{margin-left:auto;color:var(--muted);font-size:12px}.notice{display:none;padding:11px 14px;margin:0 0 18px;background:var(--orange-soft);border-left:3px solid var(--orange);color:#70472e}.notice.show{display:block}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);box-shadow:var(--shadow);margin-bottom:28px}.stat{min-height:112px;background:var(--white);padding:16px 19px}.stat-label{font-size:12px;color:var(--muted);font-weight:650}.stat-value{font-variant-numeric:tabular-nums;font-size:29px;line-height:1.2;font-weight:720;margin:8px 0 4px}.stat-foot{font-size:11px;color:var(--muted)}.stat.good .stat-value{color:var(--green)}.stat.warn .stat-value{color:var(--orange)}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin:27px 0 13px}.section-head h2{font-size:18px;margin:0;font-weight:720}.section-head p{font-size:12px;color:var(--muted);margin:3px 0 0}.hint{font-size:11px;color:var(--muted)}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.chart-panel{background:var(--white);border:1px solid var(--line);border-radius:5px;padding:15px 16px 10px;min-width:0;box-shadow:var(--shadow)}.chart-panel.wide{grid-column:1/-1}.chart-title{display:flex;justify-content:space-between;align-items:start;gap:10px}.chart-title h3{font-size:14px;margin:0;font-weight:700}.chart-title span{font-size:11px;color:var(--muted);white-space:nowrap}.chart-svg{display:block;width:100%;height:auto;min-height:174px;overflow:visible}.chart-svg text{font-family:inherit;fill:#71817f;font-size:11px}.chart-svg .grid{stroke:#e8eeeb;stroke-width:1}.chart-svg .axis{stroke:#aebbb6;stroke-width:1}.chart-svg .series{fill:none;stroke:var(--green);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.chart-svg .series.secondary{stroke:var(--blue)}.chart-svg .dot{fill:var(--white);stroke:var(--green);stroke-width:2}.chart-svg .dot.secondary{stroke:var(--blue)}.chart-svg .threshold{stroke:var(--orange);stroke-width:1.5;stroke-dasharray:5 4}.chart-svg .threshold-label{fill:var(--orange);font-weight:650}.chart-legend{display:flex;gap:15px;flex-wrap:wrap;padding:2px 4px 5px;color:var(--muted);font-size:11px}.legend-key{display:inline-flex;align-items:center;gap:6px}.swatch{width:15px;height:2px;background:var(--green)}.swatch.blue{background:var(--blue)}.swatch.orange{height:0;border-top:2px dashed var(--orange);background:none}.phase-list{display:grid;gap:10px;padding:15px 2px 6px}.phase-row{display:grid;grid-template-columns:110px minmax(80px,1fr) 50px;align-items:center;gap:10px;font-size:12px}.phase-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#435452}.phase-track{height:8px;background:#edf1ee;border-radius:2px;overflow:hidden}.phase-fill{height:100%;background:var(--orange);border-radius:2px}.phase-count{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}.run-tools{display:flex;align-items:center;gap:7px}.filter-button{border:1px solid var(--line);background:white;border-radius:4px;padding:6px 10px;font-size:12px;color:var(--muted)}.filter-button.active{background:var(--ink);border-color:var(--ink);color:white}.table-wrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:5px;box-shadow:var(--shadow)}table{width:100%;border-collapse:collapse;min-width:900px}th{text-align:left;background:#f0f4f1;color:var(--muted);font-size:11px;font-weight:700;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:11px 12px;border-bottom:1px solid #edf1ee;font-size:12px;vertical-align:middle}tbody tr:last-child td{border-bottom:0}.mono{font-family:ui-monospace,Consolas,monospace;font-size:11px}.num{font-variant-numeric:tabular-nums}.status{display:inline-flex;align-items:center;gap:6px;border-radius:3px;padding:3px 7px;font-size:11px;font-weight:700;white-space:nowrap}.status::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}.status.success{color:var(--green);background:var(--green-soft)}.status.failed{color:var(--red);background:var(--red-soft)}.status.timeout{color:var(--orange);background:var(--orange-soft)}.status.other{color:var(--blue);background:var(--blue-soft)}.detail-row td{background:#fbfcfb;padding:15px 18px 18px}.detail-grid{display:grid;grid-template-columns:minmax(240px,1.1fr) minmax(210px,.9fr);gap:20px}.result-detail{margin:0 0 12px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}.detail-grid h4{margin:0 0 9px;font-size:12px}.evidence{display:flex;gap:9px;flex-wrap:wrap}.evidence a{display:block;width:145px;color:var(--muted);font-size:11px;text-decoration:none}.evidence img{width:145px;height:82px;object-fit:cover;background:#e7ece9;border:1px solid var(--line);display:block;margin-bottom:4px}.detail-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px 14px;margin:0}.detail-list div{border-bottom:1px solid var(--line);padding:4px 0}.detail-list dt{font-size:10px;color:var(--muted)}.detail-list dd{margin:2px 0 0;overflow-wrap:anywhere;font-size:11px}.detail-json{grid-column:1/-1;margin:0;max-height:180px;overflow:auto;background:#f1f4f2;border-radius:4px;padding:10px;font:11px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap}.link{color:var(--green);text-decoration:none;font-weight:650}.link:hover{text-decoration:underline}.empty{padding:32px;text-align:center;color:var(--muted)}.footer{display:flex;justify-content:space-between;gap:20px;margin-top:20px;padding-top:13px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}
@media(max-width:820px){.shell{padding:18px 16px 36px}.hero{align-items:start;flex-direction:column;padding:27px 0 20px}.generated{text-align:left}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.charts{grid-template-columns:1fr}.chart-panel.wide{grid-column:auto}.toolbar-note{width:100%;margin-left:0}.detail-grid{grid-template-columns:1fr}.detail-json{grid-column:auto}}
@media(max-width:440px){.hero h1{font-size:25px}.stats{grid-template-columns:1fr 1fr}.stat{padding:13px;min-height:100px}.stat-value{font-size:24px}.phase-row{grid-template-columns:85px minmax(60px,1fr) 36px}.section-head{align-items:start;flex-direction:column}.footer{flex-direction:column}}
</style>
</head>
<body>
<main class="shell">
  <div class="topline"><div class="brand"><span class="brand-mark"></span><span>SIMULATION / EVALUATION</span></div><span>ROBOT MANIPULATION LAB</span></div>
  <header class="hero"><div><div class="eyebrow">BOTTLE PICK & PLACE</div><h1>任务评测报告</h1><p>汇总视觉抓取运行结果，检查成功率、失败阶段和关键误差。</p></div><div class="generated">报告生成时间<br><strong id="generatedAt">--</strong></div></header>
  <section class="stats" aria-label="批次概况">
    <div class="stat"><div class="stat-label">运行记录数</div><div class="stat-value" id="runCount">--</div><div class="stat-foot">当前报告中的单轮记录</div></div>
    <div class="stat good"><div class="stat-label">完整任务成功率</div><div class="stat-value" id="successRate">--</div><div class="stat-foot" id="successCount">抓取、放置、稳定验收全部通过</div></div>
    <div class="stat"><div class="stat-label">抓取验证通过</div><div class="stat-value" id="pickRate">--</div><div class="stat-foot" id="pickCoverage">已确认瓶子抬升并保持</div></div>
    <div class="stat warn"><div class="stat-label">放置 XY 误差中位数</div><div class="stat-value" id="placementMedian">--</div><div class="stat-foot">仅统计有最终误差记录的运行</div></div>
  </section>

  <section>
    <div class="section-head"><div><h2>关键指标</h2><p>按运行时间排序；单轮随机样本之间的连线只用于观察波动，不代表因果关系。</p></div><span class="hint" id="chartCohort"></span></div>
    <div class="charts">
      <article class="chart-panel"><div class="chart-title"><h3>累计完整成功率</h3><span>比例 · 趋势</span></div><svg class="chart-svg" id="successChart" viewBox="0 0 720 230" role="img" aria-label="累计完整成功率折线图"></svg><div class="chart-legend"><span class="legend-key"><i class="swatch"></i>累计成功率</span></div></article>
      <article class="chart-panel"><div class="chart-title"><h3>单轮放置误差</h3><span>毫米 · 越低越好</span></div><svg class="chart-svg" id="placementChart" viewBox="0 0 720 230" role="img" aria-label="单轮放置误差折线图"></svg><div class="chart-legend"><span class="legend-key"><i class="swatch blue"></i>XY 误差</span><span class="legend-key"><i class="swatch orange"></i>验收线 15 mm</span></div></article>
      <article class="chart-panel"><div class="chart-title"><h3>单轮最大搬运滑移</h3><span>毫米 · 越低越好</span></div><svg class="chart-svg" id="slipChart" viewBox="0 0 720 230" role="img" aria-label="最大搬运滑移折线图"></svg><div class="chart-legend"><span class="legend-key"><i class="swatch"></i>最大滑移</span><span class="legend-key"><i class="swatch orange"></i>监控线 15 mm</span></div></article>
      <article class="chart-panel"><div class="chart-title"><h3>失败阶段分布</h3><span>次数</span></div><div class="phase-list" id="phaseChart"></div></article>
    </div>
  </section>

  <section>
    <div class="section-head"><div><h2>单轮记录</h2><p>点击「详情」查看失败原因、视觉证据和任务配置。</p></div><div class="run-tools" id="statusFilters"><button class="filter-button active" data-filter="all">全部</button><button class="filter-button" data-filter="success">成功</button><button class="filter-button" data-filter="failed">失败 / 超时</button></div></div>
    <div class="table-wrap"><table><thead><tr><th>运行编号</th><th>结果</th><th>终止阶段</th><th>置信度</th><th>定位误差</th><th>最大滑移</th><th>放置误差</th><th>耗时</th><th></th></tr></thead><tbody id="runTable"></tbody></table><div class="empty" id="emptyState" hidden>当前筛选没有运行记录。</div></div>
  </section>
  <footer class="footer"><span id="sourceNote"></span><span id="excludedNote"></span></footer>
</main>
<script type="application/json" id="reportData">__REPORT_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('reportData').textContent);
const $=id=>document.getElementById(id);const fmt=(v,d=1)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let statusFilter='all';
const statuses=r=>String(r.status).toLowerCase();
const isSuccess=r=>statuses(r)==='succeeded';
const isTimeout=r=>['timeout','timedout'].includes(statuses(r));
function selectedRuns(){return DATA.runs}
function setMetrics(){const runs=selectedRuns(),n=runs.length,successes=runs.filter(isSuccess).length,pickKnown=runs.filter(r=>r.pick_verified!==null&&r.pick_verified!==undefined),picked=pickKnown.filter(r=>r.pick_verified).length,errors=runs.map(r=>r.placement_error_mm).filter(Number.isFinite).sort((a,b)=>a-b);const med=errors.length?(errors.length%2?errors[(errors.length-1)/2]:(errors[errors.length/2-1]+errors[errors.length/2])/2):null;
 $('runCount').textContent=n;$('successRate').textContent=n?`${(100*successes/n).toFixed(1)}%`:'—';$('successCount').textContent=`${successes} / ${n} 完整成功`;$('pickRate').textContent=pickKnown.length?`${(100*picked/pickKnown.length).toFixed(1)}%`:'—';$('pickCoverage').textContent=`${picked}/${pickKnown.length} 条有效记录`;$('placementMedian').textContent=med===null?'—':`${med.toFixed(1)} mm`;
 $('chartCohort').textContent=`${n} 条记录`;
}
function svgEl(name,attrs={},text=''){const e=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e;}
function drawLineChart(svgId,records,key,{limit=null,limitLabel='',percent=false,secondary=false}={}){const svg=$(svgId);svg.replaceChildren();const W=720,H=230,L=48,R=17,T=15,B=33,PW=W-L-R,PH=H-T-B;const values=records.map(r=>r[key]);const valid=values.filter(Number.isFinite);if(!valid.length){svg.append(svgEl('text',{x:W/2,y:H/2,'text-anchor':'middle'},'暂无可绘制的数据'));return}
 let max=percent?100:Math.max(...valid,limit||0);if(max===0)max=1;max=percent?100:max*1.12;const min=0;const x=i=>L+(records.length<=1?PW/2:i*PW/(records.length-1));const y=v=>T+(max-v)/(max-min)*PH;
 for(let i=0;i<=4;i++){const v=max*i/4,yy=y(v);svg.append(svgEl('line',{x1:L,y1:yy,x2:W-R,y2:yy,class:'grid'}));svg.append(svgEl('text',{x:L-8,y:yy+4,'text-anchor':'end'},percent?`${Math.round(v)}%`:fmt(v,0)));}
 svg.append(svgEl('line',{x1:L,y1:T+PH,x2:W-R,y2:T+PH,class:'axis'}));if(limit!==null){const yy=y(limit);svg.append(svgEl('line',{x1:L,y1:yy,x2:W-R,y2:yy,class:'threshold'}));svg.append(svgEl('text',{x:W-R-2,y:Math.max(T+11,yy-5),'text-anchor':'end',class:'threshold-label'},limitLabel));}
 const pts=[];records.forEach((r,i)=>{const v=r[key];if(Number.isFinite(v))pts.push([x(i),y(v),r])});if(pts.length){let d='';let pen=false;for(let i=0;i<records.length;i++){const v=records[i][key];if(!Number.isFinite(v)){pen=false;continue}d+=`${pen?' L':'M'} ${x(i)} ${y(v)}`;pen=true}svg.append(svgEl('path',{d,class:`series${secondary?' secondary':''}`}));for(const [cx,cy,r] of pts){const c=svgEl('circle',{cx,cy,r:4,class:`dot${secondary?' secondary':''}`});c.append(svgEl('title',{},`${r.run_id}: ${fmt(r[key],1)}${percent?'%':' mm'}`));svg.append(c)}}
 const ticks=[0,Math.floor((records.length-1)/2),records.length-1].filter((v,i,a)=>records.length&&a.indexOf(v)===i);for(const i of ticks){svg.append(svgEl('text',{x:x(i),y:H-8,'text-anchor':i===0?'start':i===records.length-1?'end':'middle'},`第 ${i+1} 轮`));}}
function drawPhases(runs){const counts=new Map();for(const r of runs){if(!isSuccess(r)){const p=r.phase||'Unknown';counts.set(p,(counts.get(p)||0)+1)}}const sorted=[...counts].sort((a,b)=>b[1]-a[1]);const host=$('phaseChart');host.replaceChildren();if(!sorted.length){host.innerHTML='<div class="empty">当前筛选没有失败运行。</div>';return}const max=sorted[0][1];for(const [name,count]of sorted){const row=document.createElement('div');row.className='phase-row';row.innerHTML=`<span class="phase-name" title="${esc(name)}">${esc(name)}</span><div class="phase-track"><div class="phase-fill" style="width:${100*count/max}%"></div></div><span class="phase-count">${count} 次</span>`;host.append(row)}}
function statusClass(r){if(isSuccess(r))return'success';if(isTimeout(r))return'timeout';if(statuses(r)==='failed')return'failed';return'other'}function statusText(r){if(isSuccess(r))return'完整成功';if(isTimeout(r))return'超时';if(statuses(r)==='failed')return'失败';return r.status||'未知'}
function detailHtml(r){const evidence=Object.entries({RGB:r.files.rgb,Mask:r.files.mask,预测:r.files.prediction}).filter(([,v])=>v);const imageHtml=evidence.length?evidence.map(([label,url])=>`<a href="${esc(url)}" target="_blank"><img loading="lazy" src="${esc(url)}" alt="${esc(label)}"><span>${esc(label)} · 打开原图</span></a>`).join(''):'<span class="hint">无视觉文件</span>';const conf=r.confidence==null?'—':`${(100*r.confidence).toFixed(1)}%`;const detail=JSON.stringify({status:r.status,phase:r.phase,detail:r.detail,confidence:r.confidence,pose_error_mm:r.pose_error_mm,max_lift_mm:r.max_lift_mm,sim_seconds:r.sim_seconds,wall_seconds:r.wall_seconds,configuration:r.configuration,events:r.events},null,2);return `<div class="detail-grid"><div><h4>本轮视觉证据</h4><div class="evidence">${imageHtml}</div></div><div><h4>本轮结果</h4><p class="result-detail">${esc(r.detail||"无附加说明")}</p><dl class="detail-list"><div><dt>模型置信度</dt><dd>${conf}</dd></div><div><dt>仿真定位误差</dt><dd>${fmt(r.pose_error_mm)} mm</dd></div><div><dt>最大抬升</dt><dd>${fmt(r.max_lift_mm)} mm</dd></div><div><dt>仿真耗时</dt><dd>${fmt(r.sim_seconds,2)} s</dd></div><div><dt>结果报告</dt><dd>${r.result_file?`<a class="link" href="${esc(r.result_file)}" target="_blank">打开 JSON</a>`:'—'}</dd></div><div><dt>轮次编号</dt><dd class="mono">${esc(r.run_id)}</dd></div></dl></div><pre class="detail-json">${esc(detail)}</pre></div>`}
function renderTable(){const runs=selectedRuns().slice().reverse().filter(r=>statusFilter==='all'||(statusFilter==='success'?isSuccess(r):!isSuccess(r)));const body=$('runTable');body.replaceChildren();$('emptyState').hidden=!!runs.length;for(const r of runs){const tr=document.createElement('tr');const place=Number.isFinite(r.placement_error_mm)?`${fmt(r.placement_error_mm)} mm`:'—';const slip=Number.isFinite(r.slip_mm)?`${fmt(r.slip_mm)} mm`:'—';const conf=r.confidence==null?'—':`${(100*r.confidence).toFixed(1)}%`;const elapsed=Number.isFinite(r.sim_seconds)?`${fmt(r.sim_seconds,1)} s`:(Number.isFinite(r.wall_seconds)?`${fmt(r.wall_seconds,1)} s wall`:'—');tr.innerHTML=`<td class="mono">${esc(r.run_id)}</td><td><span class="status ${statusClass(r)}">${statusText(r)}</span></td><td>${esc(r.phase||'—')}</td><td class="num">${conf}</td><td class="num">${Number.isFinite(r.pose_error_mm)?`${fmt(r.pose_error_mm)} mm`:"—"}</td><td class="num">${slip}</td><td class="num">${place}</td><td class="num">${elapsed}</td><td><button class="filter-button detail-toggle" aria-expanded="false">详情</button></td>`;const dr=document.createElement('tr');dr.className='detail-row';dr.hidden=true;const td=document.createElement('td');td.colSpan=9;td.innerHTML=detailHtml(r);dr.append(td);tr.querySelector('button').addEventListener('click',e=>{dr.hidden=!dr.hidden;e.currentTarget.setAttribute('aria-expanded',String(!dr.hidden));e.currentTarget.textContent=dr.hidden?'详情':'收起'});body.append(tr,dr)}}
function render(){const runs=selectedRuns();setMetrics();const cum=[];let successes=0;runs.forEach((r,i)=>{if(isSuccess(r))successes++;cum.push({...r,cumulative_rate:100*successes/(i+1)})});drawLineChart('successChart',cum,'cumulative_rate',{percent:true});drawLineChart('placementChart',runs,'placement_error_mm',{limit:DATA.placement_limit_mm,limitLabel:`当前参考 ${DATA.placement_limit_mm} mm`,secondary:true});drawLineChart('slipChart',runs,'slip_mm',{limit:DATA.slip_limit_mm,limitLabel:`当前参考 ${DATA.slip_limit_mm} mm`});drawPhases(runs);renderTable()}
function init(){$('generatedAt').textContent=new Date(DATA.generated_at).toLocaleString('zh-CN');$('sourceNote').textContent=`数据来源：${DATA.source}`;$('excludedNote').textContent=DATA.ignored_run_ids.length?`已排除 ${DATA.ignored_run_ids.length} 条调试或不可读记录`:'';document.querySelectorAll('.filter-button[data-filter]').forEach(b=>b.addEventListener('click',()=>{statusFilter=b.dataset.filter;document.querySelectorAll('.filter-button[data-filter]').forEach(x=>x.classList.toggle('active',x===b));renderTable()}));render()}
init();
</script>
</body>
</html>'''.replace("__REPORT_DATA__", payload)


def build_report(run_root=DEFAULT_RUN_ROOT, output=DEFAULT_OUTPUT, exclude_run_ids=()):
    run_root = Path(run_root).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run folder does not exist: {run_root}")
    output = Path(output).resolve()
    exclusions = _read_json(DEFAULT_EXCLUSIONS) or {}
    excluded_ids = set(exclusions) | set(exclude_run_ids)
    runs, excluded = load_runs(run_root, output.parent, excluded_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(runs, excluded, run_root), encoding="utf-8")
    return output, len(runs), len(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-run-id", action="append", default=[],
                        help="Exclude a known debug/invalid run; can be repeated")
    args = parser.parse_args()
    output, included, excluded = build_report(
        run_root=args.run_root, output=args.output, exclude_run_ids=args.exclude_run_id)
    print(f"Generated {output}")
    print(f"Included {included} runs; excluded {excluded} runs")


if __name__ == "__main__":
    main()
