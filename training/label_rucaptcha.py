# -*- coding: utf-8 -*-
"""Разметка трудных капч через RuCaptcha (ключ — переменная окружения RUCAPTCHA_KEY).
Вход JSONL: {"file": путь, "fam": dig|kchr|cyr|23os|mir, "pred": ответ модели или "", "mode": "val"|"agree"}.
  val   — метка независима от модели: два ответа RuCaptcha, при расхождении третий, берём большинство;
  agree — первый ответ RuCaptcha сверяем с моделью; совпал — метка; нет — второй ответ, большинство из трёх.
Выход JSONL (дописывается, повторный запуск пропускает готовые): file, fam, label, ans[], pred, cost.
Бюджет — --budget рублей на ЭТОТ запуск (по полю cost из getTaskResult, иначе 0,044 за задание)."""
import sys, os, re, json, time, base64, threading, argparse, collections
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.rucaptcha.com/"
HOMO = str.maketrans("aeopcyxkmthbr", "аеорсухкмтнвг")
DIG = set("0123456789")
CYR = set("0123456789абвгдежзийклмнопрстуфхцчшщъыьэюя")
lock = threading.Lock()
spent = [0.0]
tasks_n = [0]


KEY = os.environ.get("RUCAPTCHA_KEY", "")


def api(method, payload):
    payload = dict(payload, clientKey=KEY)
    req = urllib.request.Request(API + method, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def norm(ans, fam):
    a = (ans or "").strip().lower().replace(" ", "").replace("ё", "е")
    if fam in ("dig", "kchr"):
        a = a.replace("o", "0").replace("о", "0").replace("l", "1").replace("з", "3")
        return a if a and set(a) <= DIG and len(a) == 5 else None
    a = a.translate(HOMO)
    if fam == "mir" and any(c in DIG for c in a):
        return None
    lo, hi = (5, 7) if fam == "mir" else (4, 6)
    return a if a and set(a) <= CYR and lo <= len(a) <= hi else None


def ask(path, fam, budget):
    with lock:
        if spent[0] + 0.05 > budget:
            raise RuntimeError("budget")
        spent[0] += 0.044
        tasks_n[0] += 1
    body = base64.b64encode(open(path, "rb").read()).decode()
    task = {"type": "ImageToTextTask", "body": body, "phrase": False, "case": False}
    if fam in ("dig", "kchr"):
        task.update(numeric=1, minLength=5, maxLength=5, comment="ровно 5 цифр")
        extra = {}
    else:
        task.update(numeric=2 if fam == "mir" else 0, minLength=5 if fam == "mir" else 4, maxLength=7 if fam == "mir" else 6,
                    comment="строчные русские буквы" + ("" if fam == "mir" else " и цифры"))
        extra = {"languagePool": "rn"}
    for attempt in range(3):
        try:
            r = api("createTask", dict({"task": task}, **extra))
            if r.get("errorId"):
                if r.get("errorCode") in ("ERROR_ZERO_BALANCE",):
                    raise RuntimeError("zero balance")
                time.sleep(3)
                continue
            tid = r["taskId"]
            t0 = time.time()
            time.sleep(6)
            while time.time() - t0 < 180:
                g = api("getTaskResult", {"taskId": tid})
                if g.get("errorId"):
                    break
                if g.get("status") == "ready":
                    c = float(g.get("cost") or 0.044)
                    with lock:
                        spent[0] += c - 0.044
                    return norm((g.get("solution") or {}).get("text"), fam), c
                time.sleep(3)
        except RuntimeError:
            raise
        except Exception:
            time.sleep(3)
    return None, 0.0


def majority(vals):
    c = collections.Counter(v for v in vals if v)
    if not c:
        return None
    v, n = c.most_common(1)[0]
    return v if n >= 2 else None


def work(item, budget):
    fam, path, pred, mode = item["fam"], item["file"], item.get("pred") or "", item.get("mode", "agree")
    ans, cost = [], 0.0
    try:
        a1, c = ask(path, fam, budget); ans.append(a1); cost += c
        if mode == "agree" and a1 and a1 == pred:
            return dict(item, label=a1, ans=ans, cost=round(cost, 4))
        a2, c = ask(path, fam, budget); ans.append(a2); cost += c
        if mode == "agree":
            lab = majority([a1, a2, pred])
            return dict(item, label=lab, ans=ans, cost=round(cost, 4))
        if a1 and a1 == a2:
            return dict(item, label=a1, ans=ans, cost=round(cost, 4))
        a3, c = ask(path, fam, budget); ans.append(a3); cost += c
        return dict(item, label=majority([a1, a2, a3]), ans=ans, cost=round(cost, 4))
    except RuntimeError as e:
        return dict(item, label=None, ans=ans, cost=round(cost, 4), stop=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--threads", type=int, default=40)
    a = ap.parse_args()
    if not KEY:
        raise SystemExit("нет RUCAPTCHA_KEY")
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                if "stop" not in r:
                    done.add(r["file"])
            except Exception:
                pass
    items = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    items = [i for i in items if i["file"] not in done]
    print("к разметке", len(items), "уже", len(done), "бюджет", a.budget, flush=True)
    fo = open(a.out, "a", encoding="utf-8")
    stats = collections.Counter()
    t0 = time.time()
    stop = threading.Event()

    def job(it):
        if stop.is_set():
            return
        r = work(it, a.budget)
        with lock:
            if "stop" in r:
                stop.set()
                return
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            stats[(r["fam"], r["label"] is not None)] += 1
            n = sum(stats.values())
            if n % 100 == 0:
                print("%d готово, потрачено %.2f ₽ (%d заданий), %.0fs %s" % (
                    n, spent[0], tasks_n[0], time.time() - t0, dict(stats)), flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(job, items))
    print("ИТОГ потрачено %.2f ₽, заданий %d, %s" % (spent[0], tasks_n[0], dict(stats)), flush=True)


if __name__ == "__main__":
    main()
