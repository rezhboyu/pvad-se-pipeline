#!/usr/bin/env python3
"""
最終報告生成: 所有架構比較 + 音檔 + 圖表
=========================================
1. SD 架構比較 (MLP/BiLSTM/BiGRU+Attn/CNN/LinearSVM/RF) - 噪音增強訓練
2. pVAD-SE 結果
3. 生成所有模型的輸出音檔
4. 生成比較圖表
5. 整理 presentation/ 資料夾
"""

import sys, json, time, numpy as np, librosa, shutil
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.audio import read_audio, write_audio, SAMPLE_RATE

MAT_DIR = Path.home() / "Desktop" / "VOICE" / "MAT"
PROJECT_DIR = Path(__file__).resolve().parent
PRES_DIR = PROJECT_DIR / "presentation"


# ══════ 特徵 ══════
def extract_sdmfcc(audio, sr=16000, hop=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15, hop_length=hop)
    d = librosa.feature.delta(mfcc); d2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, d, d2]).T

def energy_vad(audio, hop=512, thr=0.01):
    n = (len(audio) - hop) // hop + 1
    return np.array([1 if np.sqrt(np.mean(audio[i*hop:i*hop+hop]**2)) > thr else 0 for i in range(n)], dtype=np.int64)

def add_noise(audio, snr_db):
    n = np.random.randn(len(audio)).astype(np.float32)
    sp = np.mean(audio**2)
    return audio + n * np.sqrt(sp / (10**(snr_db/10)) / (np.mean(n**2)+1e-12))


# ══════ 模型 ══════
class SD_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,2))
    def forward(self, x): return self.net(x)

class SD_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(45,128,2,batch_first=True,bidirectional=True,dropout=0.3)
        self.fc = nn.Linear(256,2); self.drop = nn.Dropout(0.3)
    def forward(self, x):
        if x.dim()==2: x = x.unsqueeze(1)
        o,_ = self.lstm(x); return self.fc(self.drop(o.squeeze(1)))

class SD_BiGRU_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(45,128,2,batch_first=True,bidirectional=True,dropout=0.3)
        self.attn = nn.MultiheadAttention(256,4,batch_first=True,dropout=0.1)
        self.fc = nn.Linear(256,2); self.drop = nn.Dropout(0.3); self.norm = nn.LayerNorm(256)
    def forward(self, x):
        if x.dim()==2: x = x.unsqueeze(1)
        o,_ = self.gru(x); a,_ = self.attn(o,o,o)
        return self.fc(self.drop(self.norm(o+a).squeeze(1)))

class SD_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.2),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),
            nn.Conv1d(128,256,3,padding=1),nn.BatchNorm1d(256),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(256,2)
    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))


# ══════ 訓練 ══════
def prepare_data():
    hs = sorted(MAT_DIR.glob("*hsuan*.wav"))
    ifs = sorted(MAT_DIR.glob("*0911*.wav"))
    fs = sorted(MAT_DIR.glob("*FEMH*.wav"))
    snrs = [5,10,15,20]
    aX, ay = [], []
    for f in hs:
        a = read_audio(str(f)); feat = extract_sdmfcc(a); vad = energy_vad(a)
        ml = min(len(feat),len(vad)); aX.append(feat[:ml]); ay.append(vad[:ml])
        for s in snrs:
            fn = extract_sdmfcc(add_noise(a,s)); ml2 = min(len(fn),ml)
            aX.append(fn[:ml2]); ay.append(vad[:ml2])
    for f in list(ifs)+list(fs):
        a = read_audio(str(f)); feat = extract_sdmfcc(a)
        aX.append(feat); ay.append(np.zeros(len(feat),dtype=np.int64))
        for s in snrs:
            fn = extract_sdmfcc(add_noise(a,s))
            aX.append(fn); ay.append(np.zeros(len(fn),dtype=np.int64))
    X = np.concatenate(aX); y = np.concatenate(ay)
    Xtr,Xv,ytr,yv = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    m = Xtr.mean(0); s = Xtr.std(0)+1e-8
    return (Xtr-m)/s, (Xv-m)/s, ytr, yv, m, s

def train_torch(model, Xtr, Xv, ytr, yv, epochs=60, lr=1e-3, name=""):
    trl = DataLoader(TensorDataset(torch.FloatTensor(Xtr),torch.LongTensor(ytr)),256,shuffle=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xv),torch.LongTensor(yv)),512)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(),lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
    ba, bs = 0, None
    for ep in range(epochs):
        model.train()
        for xb,yb in trl:
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
        model.eval(); c=t=0
        with torch.no_grad():
            for xb,yb in vl: c+=(model(xb).argmax(1)==yb).sum().item(); t+=len(yb)
        acc=c/t; sch.step(1-acc)
        if acc>ba: ba=acc; bs={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(bs)
    p = sum(pp.numel() for pp in model.parameters())
    print(f"  [{name}] val={ba:.3f} params={p:,}")
    return model, ba, p


# ══════ 場景 ══════
def build_scenarios():
    hs = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*hsuan_7.wav"))[:3]]
    ia = [read_audio(str(f)) for f in sorted(MAT_DIR.glob("*0911*_7.wav"))[:2]]
    fa = read_audio(str(sorted(MAT_DIR.glob("*FEMH_7.wav"))[0]))
    sl=int(2.5*SAMPLE_RATE); g=lambda a,i:a[i%len(a)][:sl]
    sa=np.concatenate([g(hs,0),g(ia,0),g(hs,1),g(ia,1),g(hs,2)])
    sd=np.concatenate([g(hs,0),fa[:sl],g(hs,1),fa[sl:2*sl],g(hs,2)])
    n=np.random.RandomState(42).randn(len(sa)).astype(np.float32)
    n*=np.sqrt(np.mean(sa**2)/10/(np.mean(n**2)+1e-12))
    sc=sa+n
    ol=int(3.0*SAMPLE_RATE)
    sb=np.concatenate([g(hs,0),hs[0][sl:sl+ol]+ia[0][:ol],g(hs,1)])
    labs=[("hsuan_0.0-2.5s",0,2.5,True),("interferer_2.5-5.0s",2.5,5,False),
          ("hsuan_5.0-7.5s",5,7.5,True),("interferer_7.5-10.0s",7.5,10,False),
          ("hsuan_10.0-12.5s",10,12.5,True)]
    labs_b=[("hsuan_only_0.0-2.5s",0,2.5,True),("hsuan+interferer_2.5-5.5s",2.5,5.5,True),
            ("hsuan_only_5.5-8.0s",5.5,8,True)]
    return {"scenario_a":(sa,labs),"scenario_b":(sb,labs_b),"scenario_c":(sc,labs),"scenario_d":(sd,labs)}


def main():
    print("="*80)
    print("最終報告生成")
    print("="*80)

    # 清理 presentation 資料夾
    if PRES_DIR.exists():
        shutil.rmtree(PRES_DIR)
    PRES_DIR.mkdir()
    (PRES_DIR / "audio").mkdir()
    (PRES_DIR / "charts").mkdir()

    # 準備資料
    Xtr,Xv,ytr,yv,mean,std = prepare_data()
    print(f"資料: {len(Xtr)} train, {len(Xv)} val")

    # 訓練模型
    print("\n--- 訓練 ---")
    models = {}
    for nm, cls in [("MLP",SD_MLP),("BiLSTM",SD_BiLSTM),("BiGRU_Attn",SD_BiGRU_Attn),("CNN",SD_CNN)]:
        t0=time.time()
        m,va,p = train_torch(cls(),Xtr,Xv,ytr,yv,epochs=60,name=nm)
        models[nm]={"model":m,"type":"torch","val_acc":va,"params":p,"time":time.time()-t0}

    t0=time.time()
    lsvm=LinearSVC(C=1.0,class_weight="balanced",max_iter=2000); lsvm.fit(Xtr,ytr)
    la=accuracy_score(yv,lsvm.predict(Xv))
    models["LinearSVM"]={"model":lsvm,"type":"sklearn","val_acc":la,"params":"N/A","time":time.time()-t0}
    print(f"  [LinearSVM] val={la:.3f}")

    t0=time.time()
    rf=RandomForestClassifier(100,max_depth=15,class_weight="balanced",random_state=42); rf.fit(Xtr,ytr)
    ra=accuracy_score(yv,rf.predict(Xv))
    models["RF"]={"model":rf,"type":"sklearn","val_acc":ra,"params":"N/A","time":time.time()-t0}
    print(f"  [RF] val={ra:.3f}")

    # 場景
    scenarios = build_scenarios()

    # 載入 pVAD-SE 結果
    with open(PROJECT_DIR/"test_parallel/test_report_parallel.json",encoding="utf-8") as f:
        pvad_m = json.load(f)["metrics"]

    # 測試 + 生成音檔
    print("\n--- 測試 & 生成音檔 ---")
    hop = 512; fd = hop/SAMPLE_RATE
    all_results = {}

    for mn, minfo in models.items():
        all_results[mn] = {"per_scenario":{}, "tp":0,"fn":0,"fp":0,"tn":0}
        for sc_name,(audio,labels) in scenarios.items():
            feat = (extract_sdmfcc(audio)-mean)/std
            if minfo["type"]=="torch":
                minfo["model"].eval()
                with torch.no_grad():
                    preds = minfo["model"](torch.FloatTensor(feat)).argmax(1).numpy()
            else:
                preds = minfo["model"].predict(feat)

            # 生成輸出音檔
            output = np.zeros_like(audio)
            for i in range(len(preds)):
                s=i*hop; e=min(s+hop,len(audio))
                output[s:e] = audio[s:e] * (1.0 if preds[i]==1 else 0.0)
            write_audio(str(PRES_DIR/"audio"/f"{sc_name}_{mn}_output.wav"), output)

            sc_res = {}
            for sn,ss,se,is_t in labels:
                sf=int(ss/fd); ef=min(int(se/fd),len(preds))
                tr=float(np.mean(preds[sf:ef]))
                sc_res[sn]={"target_ratio":round(tr,4),"is_target":is_t,"n_frames":ef-sf}
                if sc_name != "scenario_b":
                    t=int(tr*(ef-sf)); nt=(ef-sf)-t
                    if is_t: all_results[mn]["tp"]+=t; all_results[mn]["fn"]+=nt
                    else: all_results[mn]["fp"]+=t; all_results[mn]["tn"]+=nt
            all_results[mn]["per_scenario"][sc_name]=sc_res

        tp,fn,fp,tn = all_results[mn]["tp"],all_results[mn]["fn"],all_results[mn]["fp"],all_results[mn]["tn"]
        p=tp/(tp+fp) if(tp+fp) else 0; r=tp/(tp+fn) if(tp+fn) else 0
        f1=2*p*r/(p+r) if(p+r) else 0; acc=(tp+tn)/(tp+tn+fp+fn)
        all_results[mn].update({"accuracy":acc,"precision":p,"recall":r,"f1":f1,
                                "val_acc":minfo["val_acc"],"params":minfo["params"]})

    # 複製 pVAD-SE 音檔
    for sc in ["a","b","c","d"]:
        src = PROJECT_DIR/f"test_parallel/scenario_{sc}_mixed.wav"
        if src.exists(): shutil.copy(src, PRES_DIR/"audio"/f"scenario_{sc}_mixed.wav")
        src = PROJECT_DIR/f"test_parallel/scenario_{sc}_output.wav"
        if src.exists(): shutil.copy(src, PRES_DIR/"audio"/f"scenario_{sc}_pVAD-SE_output.wav")
        src = PROJECT_DIR/f"test_parallel/scenario_{sc}_denoised.wav"
        if src.exists(): shutil.copy(src, PRES_DIR/"audio"/f"scenario_{sc}_denoised.wav")
    src = PROJECT_DIR/"test_parallel/enrollment_hsuan.wav"
    if src.exists(): shutil.copy(src, PRES_DIR/"audio/enrollment_hsuan.wav")

    # ══════ 圖表 ══════
    print("\n--- 生成圖表 ---")
    plt.style.use("dark_background")

    # 1. F1 Score 比較圖
    fig, ax = plt.subplots(figsize=(12,5))
    mnames = sorted(all_results.keys(), key=lambda x: all_results[x]["f1"], reverse=True)
    f1s = [all_results[m]["f1"] for m in mnames]
    f1s.append(0.672)  # pVAD-SE
    mnames_plot = mnames + ["pVAD-SE"]
    colors = ["#14B8A6" if m!="pVAD-SE" else "#22C55E" for m in mnames_plot]
    bars = ax.bar(mnames_plot, f1s, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Speaker-Dependent 架構比較 (噪音增強訓練, 場景 A+C+D)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.672, color="#22C55E", linestyle="--", alpha=0.5, label="pVAD-SE baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PRES_DIR/"charts/f1_comparison.png", dpi=150, facecolor="#0F1B2D")
    plt.close()

    # 2. 場景 C 噪音對比圖
    fig, ax = plt.subplots(figsize=(12,5))
    c_segments = ["hsuan\n0-2.5s","interf\n2.5-5s","hsuan\n5-7.5s","interf\n7.5-10s","hsuan\n10-12.5s"]
    c_labels_list = ["hsuan_0.0-2.5s","interferer_2.5-5.0s","hsuan_5.0-7.5s","interferer_7.5-10.0s","hsuan_10.0-12.5s"]
    x = np.arange(len(c_segments))
    width = 0.12
    top_models = mnames[:3]  # top 3 SD models
    for i, mn in enumerate(top_models):
        vals = [all_results[mn]["per_scenario"]["scenario_c"][sn]["target_ratio"]*100 for sn in c_labels_list]
        ax.bar(x + i*width, vals, width, label=mn, alpha=0.8)
    # pVAD-SE
    pvad_vals = [pvad_m.get("scenario_c",{}).get(sn,{}).get("target_ratio",0)*100 for sn in c_labels_list]
    ax.bar(x + len(top_models)*width, pvad_vals, width, label="pVAD-SE", color="#22C55E", alpha=0.8)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(c_segments)
    ax.set_ylabel("Target Ratio %")
    ax.set_title("場景 C (噪音 SNR 10dB) - SD 模型 vs pVAD-SE", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PRES_DIR/"charts/scenario_c_comparison.png", dpi=150, facecolor="#0F1B2D")
    plt.close()

    # 3. Precision vs Recall 散點圖
    fig, ax = plt.subplots(figsize=(8,6))
    for mn in mnames:
        ax.scatter(all_results[mn]["recall"], all_results[mn]["precision"],
                   s=100, label=f"{mn} (F1={all_results[mn]['f1']:.3f})", zorder=5)
    ax.scatter(0.598, 0.768, s=150, marker="*", color="#22C55E", label="pVAD-SE (F1=0.672)", zorder=5)
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision vs Recall", fontsize=14, fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    # F1 contours
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_range = np.linspace(0.01,1,100)
        p_range = f1_val*r_range/(2*r_range-f1_val)
        valid = (p_range>0)&(p_range<=1)
        ax.plot(r_range[valid], p_range[valid], "--", alpha=0.3, color="gray")
        ax.text(0.95, f1_val*0.95/(2*0.95-f1_val)+0.02, f"F1={f1_val}", fontsize=7, alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PRES_DIR/"charts/precision_recall.png", dpi=150, facecolor="#0F1B2D")
    plt.close()

    # 4. 參數量 vs F1 圖
    fig, ax = plt.subplots(figsize=(8,6))
    for mn in mnames:
        p = all_results[mn]["params"]
        if isinstance(p, int):
            ax.scatter(p, all_results[mn]["f1"], s=100, label=mn, zorder=5)
            ax.annotate(mn, (p, all_results[mn]["f1"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.scatter(6990260, 0.672, s=150, marker="*", color="#22C55E", label="pVAD-SE (6.9M+83K)", zorder=5)
    ax.annotate("pVAD-SE", (6990260, 0.672), textcoords="offset points", xytext=(5,5), fontsize=8, color="#22C55E")
    ax.set_xlabel("Parameters", fontsize=12); ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("模型大小 vs F1 Score", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PRES_DIR/"charts/params_vs_f1.png", dpi=150, facecolor="#0F1B2D")
    plt.close()

    # ══════ 報表 ══════
    print("\n--- 生成報表 ---")

    # 打印結果
    print(f"\n{'='*90}")
    print(f"F1 SCORE 排名")
    print(f"{'='*90}")
    print(f"{'Rank':>4s} {'Model':>12s} {'Val':>6s} {'Params':>10s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    print("-"*60)
    for rank,(mn,r) in enumerate(sorted(all_results.items(),key=lambda x:x[1]["f1"],reverse=True),1):
        ps = f"{r['params']:,}" if isinstance(r["params"],int) else r["params"]
        print(f"{rank:>4d} {mn:>12s} {r['val_acc']:5.1%} {ps:>10s} {r['accuracy']:6.1%} {r['precision']:6.1%} {r['recall']:6.1%} {r['f1']:6.3f}")
    print(f"   - {'pVAD-SE':>12s} {'N/A':>6s} {'6.9M+83K':>10s} {'65.0%':>7s} {'76.8%':>7s} {'59.8%':>7s} {'0.672':>7s}")

    # 場景 C 詳細
    print(f"\n{'='*90}")
    print(f"場景 C (噪音) TARGET RATIO")
    print(f"{'='*90}")
    for mn in sorted(all_results.keys(),key=lambda x:all_results[x]["f1"],reverse=True):
        sc_c = all_results[mn]["per_scenario"].get("scenario_c",{})
        interf_avg = np.mean([sc_c.get(s,{}).get("target_ratio",0) for s in ["interferer_2.5-5.0s","interferer_7.5-10.0s"]])
        target_avg = np.mean([sc_c.get(s,{}).get("target_ratio",0) for s in ["hsuan_0.0-2.5s","hsuan_5.0-7.5s","hsuan_10.0-12.5s"]])
        print(f"  {mn:>12s}: target_avg={target_avg:.1%}  interf_avg={interf_avg:.1%}")
    pvad_c = pvad_m.get("scenario_c",{})
    pvad_t = np.mean([pvad_c.get(s,{}).get("target_ratio",0) for s in ["hsuan_0.0-2.5s","hsuan_5.0-7.5s","hsuan_10.0-12.5s"]])
    pvad_i = np.mean([pvad_c.get(s,{}).get("target_ratio",0) for s in ["interferer_2.5-5.0s","interferer_7.5-10.0s"]])
    print(f"  {'pVAD-SE':>12s}: target_avg={pvad_t:.1%}  interf_avg={pvad_i:.1%}")

    # JSON 報表
    report = {
        "experiment": "Speaker-Dependent multi-architecture (noise-augmented) vs pVAD-SE",
        "date": "2026-03-22",
        "training": {"data": "hsuan + 0911 + FEMH (MAT)", "augmentation": "SNR 5/10/15/20 dB", "features": "45-dim SDMFCC"},
        "models": {},
        "pvad_se": {"accuracy": 0.650, "precision": 0.768, "recall": 0.598, "f1": 0.672, "params": "6.9M+83K", "rtf": 0.253},
    }
    for mn, r in sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        report["models"][mn] = {
            "val_accuracy": round(r["val_acc"],4), "params": r["params"],
            "accuracy": round(r["accuracy"],4), "precision": round(r["precision"],4),
            "recall": round(r["recall"],4), "f1": round(r["f1"],4),
            "confusion_matrix": {"tp":r["tp"],"fn":r["fn"],"fp":r["fp"],"tn":r["tn"]},
            "per_scenario": r["per_scenario"],
        }

    with open(PRES_DIR/"final_report.json","w",encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 複製 pVAD 曲線圖
    for sc in ["a","b","c","d"]:
        src = PROJECT_DIR/f"test_parallel/scenario_{sc}_pvad.png"
        if src.exists(): shutil.copy(src, PRES_DIR/f"charts/scenario_{sc}_pvad.png")

    print(f"\n所有檔案已存到: {PRES_DIR}")
    print(f"  audio/: {len(list((PRES_DIR/'audio').glob('*.wav')))} 個音檔")
    print(f"  charts/: {len(list((PRES_DIR/'charts').glob('*')))} 個圖表")
    print(f"  final_report.json")


if __name__ == "__main__":
    main()
